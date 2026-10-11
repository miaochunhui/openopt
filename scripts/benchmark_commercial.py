"""Frozen CPU LP/QP commercial adapters, with an explicit license-use gate.

Discovery uses only top-level find_spec and distribution metadata. SDK imports,
model creation and license use require --authorized-run plus an externally
approved authorization-record SHA256. This runner does not establish the legal
validity of that record, install SDKs, inspect credentials or save vendor logs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openopt.certificates import check_qp_candidate  # noqa: E402

SDKS = {"gurobi": ("gurobipy", "gurobipy"), "cplex": ("cplex", "cplex"),
        "mosek": ("mosek", "Mosek")}
KEYS = ("P", "q", "A", "l", "u")
def source_files() -> tuple[str, ...]:
    return ("scripts/benchmark_commercial.py", *sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "src/openopt").rglob("*.py")))


SOURCES = source_files()
DEFAULT_ARTIFACT = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/commercial")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problem_hash(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in KEYS:
        a = np.ascontiguousarray(arrays[key])
        h.update(json.dumps([key, a.shape, a.dtype.str]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def safe_error(error: BaseException, stage: str, **coordinate: object) -> dict:
    chain = []
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        entry = {"type": name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
                 else "REDACTED_EXCEPTION_TYPE"}
        for key in ("errno", "error_code"):
            code = getattr(current, key, None)
            if type(code) is int:
                entry[key] = code
        chain.append(entry)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return {"stage": stage, **coordinate, "causes": chain,
            "details_redacted": True, "raw_vendor_output_retained": False}


def discover(solver: str) -> dict:
    module, distribution = SDKS[solver]
    row = {"module": module, "module_available": None, "version": None,
           "status": "UNVERIFIED", "licensed_run": "UNVERIFIED", "errors": []}
    try:
        row["module_available"] = importlib.util.find_spec(module) is not None
    except Exception as error:
        row["errors"].append(safe_error(error, "find_spec", solver=solver))
    try:
        version = importlib.metadata.version(distribution)
        if not isinstance(version, str) or not re.fullmatch(r"[0-9][0-9A-Za-z.!+_-]{0,127}", version):
            raise ValueError("invalid version metadata")
        row["version"] = version
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as error:
        row["errors"].append(safe_error(error, "distribution_metadata", solver=solver))
    row["reason"] = ("DISCOVERY_ERROR" if row["errors"] else "SDK_NOT_DISCOVERED"
                     if not row["module_available"] else "DISCOVERY_ONLY_NOT_LICENSED")
    return row


@contextmanager
def silence_vendor():
    # Native license diagnostics bypass Python stream redirection. Discard both
    # file descriptors as well; this is intentionally a sequential runner.
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with open(os.devnull, "w") as sink:
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def validate_problem(arrays: dict[str, np.ndarray]) -> SimpleNamespace:
    if not all(key in arrays for key in KEYS):
        raise ValueError("frozen problem requires P/q/A/l/u")
    for key in KEYS:
        if arrays[key].dtype != np.float64:
            raise ValueError("frozen arrays must be real float64")
    p = SimpleNamespace(**{key: arrays[key] for key in KEYS})
    # Independently validates shapes, bounds, finiteness, symmetry and full PSD.
    check_qp_candidate(p, np.zeros(p.q.size), np.zeros(p.l.size))
    return p


def generated_problem(n: int, seed: int, kind: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n)) / np.sqrt(n)
    P = np.zeros((n, n)) if kind == "lp" else np.eye(n) + B.T @ B
    A = np.concatenate([np.eye(n), rng.normal(size=(3, n)) / np.sqrt(n)])
    x = rng.uniform(-0.5, 0.5, n)
    ax = A @ x
    l, u, y = ax - 0.5, ax + 0.5, np.zeros(len(ax))
    u[0], y[0] = ax[0], 0.75
    l[1], y[1] = ax[1], -0.5
    l[-1], u[-1], y[-1] = ax[-1], ax[-1], 0.3
    return dict(P=P, q=-P @ x - A.T @ y, A=A, l=l, u=u,
                expected=x, witness_y=y)


def load_problem(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if not set(KEYS).issubset(saved.files):
            raise ValueError("NPZ is missing original problem arrays")
        result = {key: np.array(saved[key], copy=True) for key in KEYS}
        if ("expected" in saved.files) != ("witness_y" in saved.files):
            raise ValueError("witness must contain both expected and witness_y")
        if "expected" in saved.files:
            result.update(expected=np.array(saved["expected"], copy=True),
                          witness_y=np.array(saved["witness_y"], copy=True))
    validate_problem(result)
    return result


def split_rows(p: SimpleNamespace) -> list[tuple[int, str, float]]:
    result = []
    for i, (lower, upper) in enumerate(zip(p.l, p.u)):
        if lower == upper:
            result.append((i, "E", float(upper)))
        else:
            if np.isfinite(lower):
                result.append((i, "G", float(lower)))
            if np.isfinite(upper):
                result.append((i, "L", float(upper)))
    return result


def mapped_dual(rows: list[tuple[int, str, float]], dual: object, m: int) -> np.ndarray:
    values = np.asarray(dual, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("malformed native row duals")
    y = np.zeros(m)
    # Native minimization row duals use c - A.T*pi. Original y uses +A.T*y.
    for (row, _, _), value in zip(rows, values):
        y[row] -= value
    return y


def parameters(solver: str, args: argparse.Namespace) -> dict:
    if solver == "gurobi":
        return dict(OutputFlag=0, LogToConsole=0, Threads=args.cpu_threads, Method=2,
                    Crossover=0, Seed=args.seed % 2000000000, TimeLimit=args.time_limit,
                    BarIterLimit=args.max_iterations, BarConvTol=args.eps_abs,
                    FeasibilityTol=args.eps_abs, OptimalityTol=args.eps_abs)
    if solver == "cplex":
        return {"threads": args.cpu_threads, "lpmethod": 4, "qpmethod": 4,
                "advance": 0, "barrier.crossover": 0, "solutiontype": 2,
                "timelimit": args.time_limit,
                "barrier.limits.iteration": args.max_iterations,
                "barrier.convergetol": args.eps_abs,
                "simplex.tolerances.feasibility": args.eps_abs,
                "simplex.tolerances.optimality": args.eps_abs}
    return {"num_threads": args.cpu_threads, "optimizer": "intpnt",
            "intpnt_basis": "never", "intpnt_starting_point": "constant",
            "intpnt_max_iterations": args.max_iterations,
            "optimizer_max_time": args.time_limit,
            "intpnt_tol_pfeas": args.eps_abs, "intpnt_tol_dfeas": args.eps_abs,
            "intpnt_tol_rel_gap": args.eps_rel, "intpnt_tol_near_rel": 1.0,
            "intpnt_qo_tol_pfeas": args.eps_abs, "intpnt_qo_tol_dfeas": args.eps_abs,
            "intpnt_qo_tol_rel_gap": args.eps_rel, "intpnt_qo_tol_near_rel": 1.0}


class GurobiAdapter:
    def __init__(self, sdk, p, config):
        self.env = sdk.Env(empty=True)
        self.model = None
        try:
            self.env.setParam("OutputFlag", 0)
            self.env.setParam("LogToConsole", 0)
            self.env.start()
            self.model = sdk.Model(env=self.env)
            for key, value in config.items():
                self.model.setParam(key, value)
            self.effective = {key: self.model.getParamInfo(key)[2] for key in config}
            self.version = ".".join(str(v) for v in sdk.gurobi.version())
            self.rows = split_rows(p)
            self.m = len(p.l)
            self.x = self.model.addVars(len(p.q), lb=-sdk.GRB.INFINITY,
                                        ub=sdk.GRB.INFINITY)
            variables = [self.x[j] for j in range(len(p.q))]
            self.constraints = []
            for row, sense, bound in self.rows:
                expr = sdk.LinExpr(p.A[row].tolist(), variables)
                relation = expr == bound if sense == "E" else expr >= bound if sense == "G" else expr <= bound
                self.constraints.append(self.model.addConstr(relation))
            objective = sdk.QuadExpr()
            objective.add(sdk.LinExpr(p.q.tolist(), variables))
            for i, j in zip(*np.nonzero(p.P)):
                objective.add(0.5 * float(p.P[i, j]) * self.x[int(i)] * self.x[int(j)])
            self.model.setObjective(objective, sdk.GRB.MINIMIZE)
            self.model.update()
            self.optimal = sdk.GRB.OPTIMAL
        except Exception:
            self.close()
            raise

    def solve(self):
        self.model.reset(1)
        self.model.optimize()
        status = int(self.model.Status)
        if status != self.optimal:
            return {"optimal": False, "status_code": status, "x": None, "y": None}
        x = np.array([self.x[j].X for j in range(len(self.x))], dtype=np.float64)
        y = mapped_dual(self.rows, [c.Pi for c in self.constraints], self.m)
        return dict(optimal=True, status_code=status, x=x, y=y)

    def close(self):
        try:
            if self.model is not None:
                self.model.dispose()
        finally:
            self.env.dispose()


class CplexAdapter:
    def __init__(self, sdk, p, config):
        self.model = sdk.Cplex()
        try:
            for setter in (self.model.set_log_stream, self.model.set_error_stream,
                           self.model.set_warning_stream, self.model.set_results_stream):
                setter(None)
            self.effective = {}
            for key, value in config.items():
                param = self.model.parameters
                for part in key.split("."):
                    param = getattr(param, part)
                param.set(value)
                self.effective[key] = param.get()
            self.version = self.model.get_version()
            self.rows = split_rows(p)
            self.m = len(p.l)
            self.model.objective.set_sense(self.model.objective.sense.minimize)
            n = len(p.q)
            self.model.variables.add(obj=p.q.tolist(), lb=[-sdk.infinity] * n,
                                     ub=[sdk.infinity] * n)
            self.model.objective.set_quadratic([
                sdk.SparsePair(ind=np.flatnonzero(row).tolist(),
                               val=row[np.flatnonzero(row)].tolist()) for row in p.P
            ])
            if self.rows:
                self.model.linear_constraints.add(
                    lin_expr=[sdk.SparsePair(ind=list(range(n)), val=p.A[row].tolist())
                              for row, _, _ in self.rows],
                    senses="".join(sense for _, sense, _ in self.rows),
                    rhs=[bound for _, _, bound in self.rows])
        except Exception:
            self.close()
            raise

    def solve(self):
        # advance=0 disables reuse of prior basis or numerical starting points.
        self.model.solve()
        status = int(self.model.solution.get_status())
        if status != self.model.solution.status.optimal:
            return dict(optimal=False, status_code=status, x=None, y=None)
        return dict(optimal=True, status_code=status,
                    x=np.asarray(self.model.solution.get_values(), dtype=np.float64),
                    y=mapped_dual(self.rows, self.model.solution.get_dual_values(), self.m))

    def close(self):
        self.model.end()


class MosekAdapter:
    def __init__(self, sdk, p, config):
        self.sdk, self.env, self.task = sdk, sdk.Env(), None
        try:
            self.task = self.env.Task(0, 0)
            t = self.task
            enums = {"optimizer": sdk.optimizertype.intpnt,
                     "intpnt_basis": sdk.basindtype.never,
                     "intpnt_starting_point": sdk.startpointtype.constant}
            self.effective = {}
            for key, value in config.items():
                if key in enums or type(value) is int:
                    param = getattr(sdk.iparam, key)
                    t.putintparam(param, int(enums.get(key, value)))
                    self.effective[key] = int(t.getintparam(param))
                else:
                    param = getattr(sdk.dparam, key)
                    t.putdouparam(param, value)
                    self.effective[key] = float(t.getdouparam(param))
            self.version = ".".join(str(v) for v in self.env.getversion())
            self.n, self.m = len(p.q), len(p.l)
            t.appendvars(self.n)
            t.appendcons(self.m)
            t.putobjsense(sdk.objsense.minimize)
            t.putclist(list(range(self.n)), p.q.tolist())
            t.putvarboundlist(list(range(self.n)), [sdk.boundkey.fr] * self.n,
                              [0.0] * self.n, [0.0] * self.n)
            bounds, lowers, uppers = [], [], []
            for lower, upper in zip(p.l, p.u):
                lf, uf = np.isfinite(lower), np.isfinite(upper)
                bounds.append(sdk.boundkey.fx if lower == upper else sdk.boundkey.ra
                              if lf and uf else sdk.boundkey.lo if lf else sdk.boundkey.up
                              if uf else sdk.boundkey.fr)
                lowers.append(float(lower) if lf else 0.0)
                uppers.append(float(upper) if uf else 0.0)
            t.putconboundlist(list(range(self.m)), bounds, lowers, uppers)
            i, j = np.nonzero(p.A)
            t.putaijlist(i.tolist(), j.tolist(), p.A[i, j].tolist())
            i, j = np.nonzero(np.tril(p.P))
            if len(i):
                t.putqobj(i.tolist(), j.tolist(), p.P[i, j].tolist())
        except Exception:
            self.close()
            raise

    def solve(self):
        self.task.optimize()
        status = self.task.getsolsta(self.sdk.soltype.itr)
        if status != self.sdk.solsta.optimal:
            return dict(optimal=False, status_code=int(status), x=None, y=None)
        x, y = [0.0] * self.n, [0.0] * self.m
        self.task.getxx(self.sdk.soltype.itr, x)
        self.task.gety(self.sdk.soltype.itr, y)
        return dict(optimal=True, status_code=int(status),
                    x=np.array(x, dtype=np.float64), y=-np.array(y, dtype=np.float64))

    def close(self):
        try:
            if self.task is not None:
                self.task.dispose()
        finally:
            self.env.dispose()


def authorization(args: argparse.Namespace) -> dict:
    if type(args.authorized_run) is not bool:
        raise ValueError("authorization flag must be boolean")
    pin = args.authorization_sha256
    if args.authorized_run is True:
        if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin):
            raise ValueError("authorized run requires an approved record SHA256")
    elif pin is not None:
        raise ValueError("authorization record requires explicit authorized-run")
    return {"explicit_authorized_run": args.authorized_run, "record_sha256": pin,
            "record_is_legal_authorization": False, "legal_validity": "UNVERIFIED",
            "operator_external_approval_attested": args.authorized_run is True}


def import_sdk(solver: str, args: argparse.Namespace):
    authorization(args)
    if args.authorized_run is not True:
        raise PermissionError("commercial SDK import requires explicit authorized-run")
    return importlib.import_module(SDKS[solver][0])


def create_adapter(solver, sdk, p, config):
    return {"gurobi": GurobiAdapter, "cplex": CplexAdapter, "mosek": MosekAdapter}[solver](sdk, p, config)


def cpu_hardware() -> dict:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    quota = Path("/sys/fs/cgroup/cpu.max")
    quota_value = quota.read_text().strip() if quota.is_file() else None
    if quota_value is not None and not re.fullmatch(r"(max|[0-9]+) [0-9]+", quota_value):
        raise ValueError("unexpected CPU quota format")
    return dict(device="cpu", architecture=platform.machine(), cpu_model=model,
                logical_cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                cgroup_cpu_max=quota_value, hostname_collected=False,
                credential_environment_collected=False)


def provenance() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if commit.returncode or not re.fullmatch(r"[a-f0-9]{40}\n?", commit.stdout):
        raise RuntimeError("source commit discovery failed")
    return {"git_commit": commit.stdout.strip(),
            "sha256": {name: file_hash(ROOT / name) for name in source_files()},
            "python": platform.python_version(), "numpy": np.__version__,
            "timing_clock": "time.perf_counter", "cpu_clock": "time.process_time"}


def write_json(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def host_values(value: object) -> list | None:
    if value is None:
        return None
    a = np.asarray(value, dtype=np.float64)
    return np.where(np.isfinite(a), a, None).tolist()


def checked_effective(config: dict, effective: dict) -> dict:
    if not isinstance(effective, dict) or set(effective) != set(config):
        raise ValueError("effective parameters are incomplete")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in effective.values()):
        raise ValueError("effective parameters must be finite public numeric settings")
    if any(type(config[key]) in (int, float) and config[key] != effective[key] for key in config):
        raise ValueError("effective numerical parameter differs from requested setting")
    return dict(effective)


def run_arm(solver, args, arrays, arm, output, report):
    stage, mode, repeat = "sdk_import", "cold", 0
    sdk, workspace = None, None
    config = arm["parameters"]
    try:
        with silence_vendor():
            sdk = import_sdk(solver, args)
        arm["sdk_imported"] = True
    except Exception as error:
        arm["failures"].append(safe_error(error, stage, solver=solver, mode=mode, repeat=repeat))
        arm["status"] = "UNVERIFIED"
        arm["reason"] = "SDK_IMPORT_FAILED_LICENSE_UNVERIFIED"
        return
    for mode in ("cold", "prepared"):
        target = arm["modes"][mode]
        for repeat in range(args.repeats):
            sample = {"repeat": repeat, "verdict": "RUNNING", "solve_attempted": False,
                      "host_x": None, "host_y": None, "certificate": None}
            target["samples"].append(sample)
            stage = "prepared_setup" if mode == "prepared" and workspace is None else "model_setup"
            start, cpu_start = time.perf_counter(), time.process_time()
            try:
                with silence_vendor():
                    if mode == "cold" or workspace is None:
                        adapter_arrays = {key: arrays[key].copy() for key in KEYS}
                        p = validate_problem(adapter_arrays)
                        arm["license_use_attempted"] = True
                        workspace = create_adapter(solver, sdk, p, config)
                        if not isinstance(workspace.version, str) or not re.fullmatch(
                                r"[0-9][0-9A-Za-z.!+_-]{0,127}", workspace.version):
                            raise ValueError("invalid runtime SDK version")
                        effective = checked_effective(config, workspace.effective)
                        if mode == "prepared":
                            target["setup_once_s"] = time.perf_counter() - start
                            start, cpu_start = time.perf_counter(), time.process_time()
                    setup_end = time.perf_counter()
                    stage = "solve_host_result"
                    sample["solve_attempted"] = True
                    solved = workspace.solve()
                    sample["host_x"] = host_values(solved["x"])
                    sample["host_y"] = host_values(solved["y"])
                    sample["solver_status_code"] = solved["status_code"]
                    sample["strict_optimal_status"] = solved["optimal"]
                    end = time.perf_counter()
                sample.update(wall_s=end - start, cpu_s=time.process_time() - cpu_start,
                              setup_s=setup_end - start, solve_host_result_s=end - setup_end,
                              runtime_version=workspace.version, effective_parameters=effective,
                              effective_parameters_sha256=digest(effective))
                arm["licensed_run"] = "EXECUTED"
                stage = "frozen_input_check"
                if problem_hash(adapter_arrays) != report["problem"]["content_sha256"]:
                    raise ValueError("adapter mutated its original-coordinate model input")
                stage = "original_certificate"
                if not solved["optimal"]:
                    raise ValueError("native solver did not return strict optimal status")
                sample["certificate"] = check_qp_candidate(
                    SimpleNamespace(**{key: arrays[key] for key in KEYS}), solved["x"], solved["y"],
                    atol=args.eps_abs, rtol=args.eps_rel).to_dict()
                if not sample["certificate"]["passed"]:
                    raise ValueError("independent original-coordinate certificate failed")
                if problem_hash(arrays) != report["problem"]["content_sha256"]:
                    raise ValueError("adapter mutated original problem")
                sample["verdict"] = "PASS"
                if mode == "cold":
                    stage = "resource_close"
                    with silence_vendor():
                        workspace.close()
                    workspace = None
            except Exception as error:
                sample["verdict"] = "FAIL"
                sample.setdefault("wall_s", time.perf_counter() - start)
                sample.setdefault("cpu_s", time.process_time() - cpu_start)
                failure = safe_error(error, stage, solver=solver, mode=mode, repeat=repeat)
                sample["failure"] = failure
                arm["failures"].append(failure)
                target["status"] = "FAIL"
                arm["status"] = "FAIL" if arm["licensed_run"] == "EXECUTED" else "UNVERIFIED"
                arm["reason"] = "ATTEMPT_FAILED_NO_RETRY"
                if workspace is not None:
                    try:
                        with silence_vendor():
                            workspace.close()
                    except Exception as close_error:
                        arm["failures"].append(safe_error(close_error, "resource_close", solver=solver,
                                                         mode=mode, repeat=repeat))
                write_json(output / "partial.json", report)
                return
            write_json(output / "partial.json", report)
        target["status"] = "PASS"
        values = [s["wall_s"] for s in target["samples"]]
        target["latency"] = {"count": len(values), "p50_s": float(np.percentile(values, 50)),
                             "p95_s": float(np.percentile(values, 95)), "samples_s": values}
    try:
        with silence_vendor():
            workspace.close()
    except Exception as error:
        arm["failures"].append(safe_error(error, "resource_close", solver=solver,
                                         mode="prepared", repeat=args.repeats - 1))
        arm["status"], arm["reason"] = "FAIL", "RESOURCE_CLOSE_FAILED"
        return
    arm["status"], arm["reason"] = "PASS", "MATCHED_LICENSED_NUMERICAL_RUN"


def finish(report: dict) -> None:
    arms = list(report["solvers"].values())
    failures = [failure for arm in arms for failure in arm["discovery"]["errors"] + arm["failures"]]
    samples = [s for arm in arms for mode in arm["modes"].values() for s in mode["samples"]]
    report["counts"] = dict(planned=len(arms) * 2 * report["repeats"], attempted=len(samples),
                            solve_attempted=sum(s["solve_attempted"] for s in samples),
                            passed=sum(s["verdict"] == "PASS" for s in samples),
                            failed=sum(s["verdict"] == "FAIL" for s in samples))
    passed = all(arm["status"] == "PASS" for arm in arms)
    report["verdict"] = "FAIL" if failures else "PASS" if passed else "UNVERIFIED"
    report["qualification"] = "PASS" if passed and not failures else "UNVERIFIED"
    report["collection_status"] = "FAIL" if failures else "COMPLETE"
    report["exit_code"] = 1 if failures else 0
    report["failures"] = failures
    report["finished_at"] = datetime.now(timezone.utc).isoformat()


def check_result(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema_version"] != 1 or report["verdict"] not in ("PASS", "FAIL", "UNVERIFIED"):
        raise ValueError("invalid terminal report")
    if type(report["repeats"]) is not int or report["repeats"] < 3 or not report["solvers"]:
        raise ValueError("empty or invalid planned denominator")
    if type(report["authorization"]["explicit_authorized_run"]) is not bool:
        raise ValueError("authorization flag must be boolean")
    if not (math.isfinite(report["eps_abs"]) and 1e-9 <= report["eps_abs"] <= 1e-2 and
            math.isfinite(report["eps_rel"]) and 0 < report["eps_rel"] < 1):
        raise ValueError("invalid certificate tolerances")
    authorization(SimpleNamespace(authorized_run=report["authorization"]["explicit_authorized_run"],
                                  authorization_sha256=report["authorization"]["record_sha256"]))
    if set(report["source"]["sha256"]) != set(source_files()):
        raise ValueError("recorded API source membership differs")
    if report["authorization"]["legal_validity"] != "UNVERIFIED" or report["authorization"]["record_is_legal_authorization"] is not False:
        raise ValueError("an approval digest cannot establish legal validity")
    for name in source_files():
        if report["source"]["sha256"][name] != file_hash(ROOT / name):
            raise ValueError("run source differs from current source")
    frozen = path.parent / "problem.npz"
    if file_hash(frozen) != report["problem"]["npz_sha256"]:
        raise ValueError("frozen NPZ bytes differ")
    arrays = load_problem(frozen)
    p = validate_problem(arrays)
    if problem_hash(arrays) != report["problem"]["content_sha256"]:
        raise ValueError("original problem hash differs")
    if "expected" in arrays:
        witness = check_qp_candidate(p, arrays["expected"], arrays["witness_y"]).to_dict()
        if not witness["passed"] or witness != report["problem"]["witness"]:
            raise ValueError("frozen witness differs")
    if not report["hardware"]["cpu_model"] or report["hardware"]["device"] != "cpu":
        raise ValueError("CPU hardware evidence missing")
    for solver, arm in report["solvers"].items():
        if solver not in SDKS or digest(arm["parameters"]) != arm["parameters_sha256"]:
            raise ValueError("solver parameters hash differs")
        if set(arm["modes"]) != {"cold", "prepared"}:
            raise ValueError("cold and prepared evidence are both required")
        if arm["status"] not in ("PASS", "FAIL", "UNVERIFIED"):
            raise ValueError("invalid terminal arm verdict")
        if (arm["sdk_imported"] or arm["licensed_run"] == "EXECUTED") and not report["authorization"]["explicit_authorized_run"]:
            raise ValueError("unauthorized SDK execution evidence")
        for mode_name, mode in arm["modes"].items():
            if mode_name not in ("cold", "prepared") or len(mode["samples"]) > report["repeats"]:
                raise ValueError("invalid sample denominator")
            for index, sample in enumerate(mode["samples"]):
                if sample["repeat"] != index or sample["verdict"] not in ("PASS", "FAIL"):
                    raise ValueError("sample coordinate or terminal verdict differs")
                if not math.isfinite(sample["wall_s"]) or sample["wall_s"] < 0:
                    raise ValueError("invalid measured wall time")
                if not math.isfinite(sample["cpu_s"]) or sample["cpu_s"] < 0:
                    raise ValueError("invalid measured CPU time")
                if sample["verdict"] == "PASS":
                    if (not sample["solve_attempted"] or arm["licensed_run"] != "EXECUTED"
                            or sample["strict_optimal_status"] is not True):
                        raise ValueError("sample lacks licensed solve evidence")
                    certificate = check_qp_candidate(p, sample["host_x"], sample["host_y"],
                                                     atol=report["eps_abs"], rtol=report["eps_rel"]).to_dict()
                    if not certificate["passed"] or certificate != sample["certificate"]:
                        raise ValueError("original-coordinate certificate differs")
                    effective = checked_effective(arm["parameters"], sample["effective_parameters"])
                    if digest(effective) != sample["effective_parameters_sha256"]:
                        raise ValueError("effective parameter hash differs")
                    if not re.fullmatch(r"[0-9][0-9A-Za-z.!+_-]{0,127}", sample["runtime_version"]):
                        raise ValueError("runtime version evidence differs")
                    if any(not math.isfinite(sample[key]) or sample[key] < 0
                           for key in ("setup_s", "solve_host_result_s")):
                        raise ValueError("invalid timing subdivision")
                    if abs(sample["wall_s"] - sample["setup_s"] - sample["solve_host_result_s"]) > 1e-9:
                        raise ValueError("timing subdivisions differ")
                elif "failure" not in sample:
                    raise ValueError("failed attempt lacks first cause")
            if mode["status"] == "PASS":
                values = [s["wall_s"] for s in mode["samples"] if s["verdict"] == "PASS"]
                if len(values) != report["repeats"] or mode["latency"] != {
                        "count": len(values), "samples_s": values,
                        "p50_s": float(np.percentile(values, 50)),
                        "p95_s": float(np.percentile(values, 95))}:
                    raise ValueError("complete-mode latency differs")
                if mode_name == "prepared" and (mode["setup_once_s"] is None or mode["setup_once_s"] < 0):
                    raise ValueError("prepared setup evidence missing")
            elif mode["latency"] is not None:
                raise ValueError("incomplete mode cannot publish qualified latency")
        if arm["status"] == "PASS" and (not arm["sdk_imported"] or
                                         any(m["status"] != "PASS" for m in arm["modes"].values())):
            raise ValueError("arm PASS lacks complete modes")
    terminal = json.loads(json.dumps(report))
    finish(terminal)
    for key in ("counts", "verdict", "qualification", "collection_status", "exit_code", "failures"):
        if terminal[key] != report[key]:
            raise ValueError("terminal aggregation differs")
    return dict(acceptance="PASS", qualification=report["qualification"],
                verdict=report["verdict"], counts=report["counts"])


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes unknown arguments and rejected values.
        self.exit(2, "COMMERCIAL_RUN verdict=FAIL stage=argument_parse details_redacted=true\n")


def main(argv: list[str] | None = None) -> int:
    parser = PrivateArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--kind", choices=("lp", "qp"), default="qp")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solvers", nargs="+", choices=tuple(SDKS), default=list(SDKS))
    parser.add_argument("--eps-abs", type=float, default=1e-7)
    parser.add_argument("--eps-rel", type=float, default=1e-7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--authorized-run", action="store_true")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--check-result", type=Path)
    args = parser.parse_args(argv)
    try:
        approval = authorization(args)
        if args.size < 2 or args.repeats < 3 or args.cpu_threads < 1 or args.max_iterations < 1:
            raise ValueError("size >=2, repeats >=3, threads/iterations >=1 required")
        if len(set(args.solvers)) != len(args.solvers):
            raise ValueError("duplicate solver selection")
        if not (math.isfinite(args.eps_abs) and 1e-9 <= args.eps_abs <= 1e-2 and
                math.isfinite(args.eps_rel) and 0 < args.eps_rel < 1 and
                math.isfinite(args.time_limit) and args.time_limit > 0):
            raise ValueError("invalid tolerances or time limit")
        if args.check_result:
            if args.authorized_run or args.problem:
                raise ValueError("replay cannot be combined with execution")
            print(json.dumps(check_result(args.check_result), allow_nan=False))
            return 0
        root = args.artifact_dir.expanduser().resolve()
        if any(root.is_relative_to(base) for base in (ROOT, Path("/data/dev"), Path("/data/wdev"))):
            raise ValueError("artifacts must be outside source worktrees")
    except Exception as error:
        print(json.dumps({"verdict": "FAIL", "error": safe_error(error, "arguments_or_replay")}))
        return 2
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-" if args.authorized_run else "discovery-", dir=root))
    report = dict(schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
                  verdict="RUNNING", qualification="UNVERIFIED", authorization=approval,
                  repeats=args.repeats, eps_abs=args.eps_abs, eps_rel=args.eps_rel,
                  scope="single frozen CPU continuous LP/QP; no commercial ranking or GPU claim",
                  timing={"cold": "validation/model/license setup/solve/host x+y; imports excluded",
                          "prepared": "same model, no iterate warm start; reset/solve/host x+y included",
                          "setup_once": "separate from prepared samples", "warmups": 0,
                          "certificate_and_disk_io": "outside measured latency"}, solvers={})
    try:
        arrays = load_problem(args.problem) if args.problem else generated_problem(args.size, args.seed, args.kind)
        p = validate_problem(arrays)
        witness = None
        if "expected" in arrays:
            witness = check_qp_candidate(p, arrays["expected"], arrays["witness_y"]).to_dict()
            if not witness["passed"]:
                raise ValueError("invalid independent planted witness")
        np.savez(output / "problem.npz", **arrays)
        report["problem"] = dict(npz_sha256=file_hash(output / "problem.npz"),
                                  content_sha256=problem_hash(arrays), dtype="float64",
                                  variables=p.q.size, constraints=p.l.size,
                                  kind="lp" if not np.any(p.P) else "qp", witness=witness)
        report["source"], report["hardware"] = provenance(), cpu_hardware()
        for solver in args.solvers:
            config = parameters(solver, args)
            report["solvers"][solver] = dict(discovery=discover(solver), status="UNVERIFIED",
                                             licensed_run="UNVERIFIED", sdk_imported=False,
                                             license_use_attempted=False,
                                             reason="AUTHORIZATION_REQUIRED", parameters=config,
                                             parameters_sha256=digest(config), failures=[],
                                             modes={name: dict(status="UNVERIFIED", setup_once_s=None,
                                                               latency=None, samples=[])
                                                    for name in ("cold", "prepared")})
        write_json(output / "partial.json", report)
        for solver, arm in report["solvers"].items():
            if arm["discovery"]["errors"]:
                arm["reason"] = "DISCOVERY_ERROR"
            elif not arm["discovery"]["module_available"]:
                arm["reason"] = "SDK_NOT_DISCOVERED"
            elif args.authorized_run is True:
                run_arm(solver, args, arrays, arm, output, report)
        finish(report)
    except Exception as error:
        report.update(verdict="FAIL", qualification="UNVERIFIED", collection_status="FAIL",
                      exit_code=1, failures=[safe_error(error, "runner")],
                      finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "result.json", report)
    if report["failures"]:
        write_json(output / "failure.json", {"first_cause": report["failures"][0],
                                             "failures": report["failures"], "exit_code": report["exit_code"]})
    if json.loads((output / "result.json").read_text()) != report:
        raise RuntimeError("persisted report differs")
    verdict, result_path, finished = report["verdict"], output / "result.json", report["finished_at"]
    line = f"COMMERCIAL_RUN verdict={verdict} finished_at={finished} result={result_path}"
    with (output / "execution.log").open("x", encoding="utf-8") as log:
        log.write(line + "\n")
    print(line)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

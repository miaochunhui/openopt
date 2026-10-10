"""Offline, fail-closed MPS/QPS import and caller-pinned frozen replay.

The existing highspy CPU reader is mandatory; an independent conservative
record audit checks that it has not discarded model semantics. No downloads,
solves, presolve, variable shifts, or public/industrial qualification occur.
All runtime files must live outside the repository in the shared temporary area.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import traceback
from urllib.parse import urlsplit

import highspy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED_TMP = Path("/hf3fs-jd/data/tmp").resolve()
VERSION = "public-qp-import-v1"
ARRAY_NAMES = ("P", "q", "A", "l", "u", "variable_lower", "variable_upper", "offset")
SCOPE = "Imported continuous convex LP/QP; public/industrial qualification UNVERIFIED"
DEFAULT_MAX_DENSE_BYTES = 128 * 1024 * 1024
HEADERS = ("NAME", "OBJSENSE", "OBJNAME", "ROWS", "COLUMNS", "RHS", "BOUNDS",
           "QUADOBJ", "QMATRIX", "ENDATA")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"{label}: expected lowercase SHA256")
    return value


def write_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    if path.read_bytes() != raw:
        raise OSError(f"JSON write/read mismatch: {path}")


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON token: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def external(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_relative_to(SHARED_TMP) or path.is_relative_to(ROOT):
        raise ValueError(f"artifacts must be outside worktree under {SHARED_TMP}: {path}")
    return path


def declaration(entry: dict) -> dict:
    required = {"case_id", "path", "source_url", "license", "input_sha256", "origin"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise ValueError(f"declaration requires exactly {sorted(required)}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", str(entry["case_id"])):
        raise ValueError("unsafe case_id")
    if not isinstance(entry["path"], str) or not entry["path"]:
        raise ValueError("missing local input path")
    url = entry["source_url"]
    if not isinstance(url, str):
        raise ValueError("source_url must be a declared HTTP(S) URL")
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(c.isspace() for c in url)):
        raise ValueError("source_url must be HTTP(S), without credentials/query/fragment")
    if not isinstance(entry["license"], str) or not entry["license"].strip():
        raise ValueError("missing license declaration (not inferred from project license)")
    if entry["origin"] not in ("public", "synthetic"):
        raise ValueError("origin must explicitly be public or synthetic")
    _sha(entry["input_sha256"], "input_sha256")
    return dict(entry)


def select_inputs(manifest: dict, local_dir: Path) -> list[tuple[dict, Path]]:
    if set(manifest) != {"schema_version", "cases"} or manifest["schema_version"] != 1:
        raise ValueError("input manifest requires schema_version=1 and cases")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("empty corpus rejected")
    local_dir = local_dir.expanduser().resolve(strict=True)
    if not local_dir.is_dir() or local_dir.is_relative_to(ROOT):
        raise ValueError("local input directory must be an explicit external directory")
    result, ids, paths = [], set(), set()
    for item in cases:
        item = declaration(item)
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input path must be relative and inside local directory")
        path = (local_dir / relative).resolve(strict=True)
        if not path.is_relative_to(local_dir) or not path.is_file():
            raise ValueError(f"input escaped local directory or is not a file: {relative}")
        if path.suffix.lower() not in (".mps", ".qps"):
            raise ValueError(f"only uncompressed MPS/QPS inputs supported: {relative}")
        if item["case_id"] in ids or path in paths:
            raise ValueError("duplicate case_id or input path")
        ids.add(item["case_id"])
        paths.add(path)
        result.append((item, path))
    return result


def _number(token: str, coordinate: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"{coordinate}: invalid number {token!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{coordinate}: nonfinite input")
    return value


def _put(mapping: dict, key: object, value: object, coordinate: str) -> None:
    if key in mapping:
        raise ValueError(f"{coordinate}: duplicate record {key!r} unsupported")
    mapping[key] = value


def audit_records(raw: bytes, max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES) -> dict:
    """Whitelist full records; preserve coordinates and audit reader results.

    RANGES is deliberately unsupported, even when a particular range could be
    represented. Multiple RHS/bound sets and ambiguous quadratic conventions
    are refused, never silently chosen. This is not a general MPS parser.
    """
    if isinstance(max_dense_bytes, bool) or not isinstance(max_dense_bytes, int) or max_dense_bytes < 1:
        raise ValueError("max_dense_bytes must be a positive integer")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("only ASCII uncompressed MPS/QPS supported") from error
    rows, columns, rhs, bound_records, quadratic = {}, {}, {}, {}, {}
    seen, sets = set(), {}
    section, sense, objname, name = None, "minimize", None, None
    sense_seen = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("*"):
            continue
        coordinate = f"line {line_number} section={section}"
        tokens = line.split()
        if "MARKER" in [t.strip("'\"").upper() for t in tokens]:
            raise ValueError(f"{coordinate}: integer MARKER unsupported")
        if not line[0].isspace():
            header = tokens[0].upper()
            if header not in HEADERS:
                raise ValueError(f"{coordinate}: unsupported construct {header}")
            if section == "ENDATA" or header in seen:
                raise ValueError(f"{coordinate}: duplicate/out-of-order header {header}")
            seen.add(header)
            section = header
            if header == "NAME":
                if name is not None or len(tokens) != 2 or len(seen) != 1:
                    raise ValueError(f"{coordinate}: NAME requires one name at start")
                name = tokens[1]
            elif name is None:
                raise ValueError(f"{coordinate}: missing NAME")
            elif header in ("OBJSENSE", "OBJNAME") and len(tokens) == 2:
                if header == "OBJSENSE":
                    if tokens[1].upper() not in ("MIN", "MINIMIZE", "MAX", "MAXIMIZE"):
                        raise ValueError(f"{coordinate}: invalid objective sense")
                    sense = "maximize" if tokens[1].upper().startswith("MAX") else "minimize"
                    sense_seen = True
                else:
                    objname = tokens[1]
            elif len(tokens) != 1:
                raise ValueError(f"{coordinate}: unsupported header arguments")
            continue
        if section in ("OBJSENSE", "OBJNAME"):
            if len(tokens) != 1:
                raise ValueError(f"{coordinate}: invalid objective declaration")
            if section == "OBJNAME":
                if objname is not None:
                    raise ValueError(f"{coordinate}: duplicate objective name")
                objname = tokens[0]
            else:
                if sense_seen or tokens[0].upper() not in ("MIN", "MINIMIZE", "MAX", "MAXIMIZE"):
                    raise ValueError(f"{coordinate}: invalid/duplicate objective sense")
                sense = "maximize" if tokens[0].upper().startswith("MAX") else "minimize"
                sense_seen = True
        elif section == "ROWS":
            if len(tokens) != 2 or tokens[0] not in ("N", "L", "G", "E"):
                raise ValueError(f"{coordinate}: unsupported row record")
            _put(rows, tokens[1], tokens[0], coordinate)
        elif section in ("COLUMNS", "RHS", "QUADOBJ", "QMATRIX"):
            if len(tokens) not in (3, 5):
                raise ValueError(f"{coordinate}: named records require one or two pairs")
            record = tokens[0]
            if section == "RHS":
                if "RHS" in sets and sets["RHS"] != record:
                    raise ValueError(f"{coordinate}: multiple RHS sets unsupported")
                sets["RHS"] = record
            if section == "COLUMNS":
                columns.setdefault(record, {})
            for target, token in zip(tokens[1::2], tokens[2::2]):
                value = _number(token, coordinate)
                if section == "COLUMNS":
                    _put(columns[record], target, value, coordinate)
                elif section == "RHS":
                    _put(rhs, target, value, coordinate)
                else:
                    _put(quadratic, (record, target), value, coordinate)
        elif section == "BOUNDS":
            if len(tokens) < 3:
                raise ValueError(f"{coordinate}: invalid bound record")
            kind, bound_set, variable = tokens[:3]
            if kind in ("BV", "LI", "UI", "SC", "SI"):
                raise ValueError(f"{coordinate}: integer/semicontinuous bound {kind} unsupported")
            if kind not in ("LO", "UP", "FX", "FR", "MI", "PL"):
                raise ValueError(f"{coordinate}: unsupported bound {kind}")
            arity = 4 if kind in ("LO", "UP", "FX") else 3
            if len(tokens) != arity:
                raise ValueError(f"{coordinate}: invalid bound arity")
            if "BOUNDS" in sets and sets["BOUNDS"] != bound_set:
                raise ValueError(f"{coordinate}: multiple BOUNDS sets unsupported")
            sets["BOUNDS"] = bound_set
            bound_records.setdefault(variable, {})
            _put(bound_records[variable], kind,
                 _number(tokens[3], coordinate) if arity == 4 else None, coordinate)
        else:
            raise ValueError(f"{coordinate}: unexpected data record")
    if not {"NAME", "ROWS", "COLUMNS", "ENDATA"} <= seen or not columns:
        raise ValueError("missing required MPS sections or empty model")
    if ("OBJSENSE" in seen and not sense_seen) or ("OBJNAME" in seen and objname is None):
        raise ValueError("empty objective declaration unsupported")
    if seen.intersection(("QUADOBJ", "QMATRIX")) and not quadratic:
        raise ValueError("empty quadratic section unsupported")
    if ("RHS" in seen and not rhs) or ("BOUNDS" in seen and not bound_records):
        raise ValueError("empty RHS/BOUNDS section unsupported")
    if {"QUADOBJ", "QMATRIX"} <= seen:
        raise ValueError("mixed quadratic sections unsupported")
    objectives = [row for row, kind in rows.items() if kind == "N"]
    if len(objectives) != 1 or (objname is not None and objname != objectives[0]):
        raise ValueError("exactly one matching objective N row required; free N rows unsupported")
    objective_row = objectives[0]
    variable_names = list(columns)
    row_names = [row for row in rows if row != objective_row]
    n, m = len(variable_names), len(row_names)
    # Account for original dense arrays and the bound-augmented solver view.
    estimate = 8 * (3 * n * n + 2 * m * n + 8 * n + 4 * m + 1)
    if estimate > max_dense_bytes:
        raise ValueError(f"dense import budget exceeded: estimated_bytes={estimate} limit={max_dense_bytes}")
    if set(rhs) - set(rows) or set(bound_records) - set(columns):
        raise ValueError("RHS/bounds reference unknown row/variable")
    index = {variable: i for i, variable in enumerate(variable_names)}
    P, q, A = np.zeros((n, n)), np.zeros(n), np.zeros((m, n))
    for j, variable in enumerate(variable_names):
        entries = columns[variable]
        if set(entries) - set(rows):
            raise ValueError(f"unknown COLUMNS row: {variable}")
        q[j] = entries.get(objective_row, 0)
        for i, row in enumerate(row_names):
            A[i, j] = entries.get(row, 0)
    for (left, right), value in quadratic.items():
        if left not in index or right not in index:
            raise ValueError("quadratic record references unknown variable")
        i, j = index[left], index[right]
        if "QMATRIX" in seen:
            if quadratic.get((right, left), 0) != value:
                raise ValueError("QMATRIX must supply an exactly symmetric full matrix")
            P[i, j] = value
        else:
            if i != j and (right, left) in quadratic:
                raise ValueError("QUADOBJ must supply only one triangle, not both")
            P[i, j] = P[j, i] = value
    l, u = np.full(m, -np.inf), np.full(m, np.inf)
    for i, row in enumerate(row_names):
        value = rhs.get(row, 0)
        if rows[row] in ("G", "E"):
            l[i] = value
        if rows[row] in ("L", "E"):
            u[i] = value
    lower, upper = np.zeros(n), np.full(n, np.inf)
    for variable, bounds in bound_records.items():
        j = index[variable]
        if "FX" in bounds:
            if len(bounds) != 1:
                raise ValueError("FX combined with other bounds unsupported")
            lower[j] = upper[j] = bounds["FX"]
        elif "FR" in bounds:
            if len(bounds) != 1:
                raise ValueError("FR combined with other bounds unsupported")
            lower[j], upper[j] = -np.inf, np.inf
        else:
            if ("LO" in bounds and "MI" in bounds) or ("UP" in bounds and "PL" in bounds):
                raise ValueError("conflicting bound declarations")
            lower[j] = bounds.get("LO", -np.inf if "MI" in bounds else 0)
            upper[j] = bounds.get("UP", np.inf)
            if upper[j] < 0 and "LO" not in bounds and "MI" not in bounds:
                raise ValueError("negative UP with implicit lower bound is ambiguous; declare LO/MI")
    arrays = dict(zip(ARRAY_NAMES, (P, q, A, l, u, lower, upper,
                                  np.asarray(-rhs.get(objective_row, 0)))))
    arrays = {key: np.asarray(value, dtype="<f8", order="C") for key, value in arrays.items()}
    model = {"arrays": arrays, "variable_names": variable_names, "row_names": row_names,
             "objective_row": objective_row, "objective_sense": sense, "model_name": name}
    validate_arrays(model)
    return model


def validate_arrays(model: dict) -> dict:
    arrays = model["arrays"]
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("frozen array membership mismatch")
    n, m = len(model["variable_names"]), len(model["row_names"])
    shapes = ((n, n), (n,), (m, n), (m,), (m,), (n,), (n,), ())
    if not n or len(set(model["variable_names"])) != n or len(set(model["row_names"])) != m:
        raise ValueError("empty/duplicate model coordinates")
    for name, shape in zip(ARRAY_NAMES, shapes):
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype("<f8"):
            raise ValueError(f"array shape/dtype mismatch: {name} expected={shape}/<f8")
    for name in ("P", "q", "A", "offset"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"nonfinite {name}")
    for low, high in (("l", "u"), ("variable_lower", "variable_upper")):
        lo, hi = arrays[low], arrays[high]
        if (np.isnan(lo).any() or np.isnan(hi).any() or np.isposinf(lo).any()
                or np.isneginf(hi).any() or (lo > hi).any()):
            raise ValueError(f"invalid bounds: {low}/{high}")
    P = arrays["P"]
    if not np.array_equal(P, P.T):
        raise ValueError("nonsymmetric Hessian")
    if model["objective_sense"] not in ("minimize", "maximize"):
        raise ValueError("unknown objective sense")
    sign = 1 if model["objective_sense"] == "minimize" else -1
    smallest = float(np.linalg.eigvalsh(sign * P).min())
    # No projection, clipping, tolerance widening, or silent convexification.
    if smallest < 0:
        raise ValueError(f"nonconvex or numerically unverified Hessian: min_eigenvalue={smallest}")
    return {"method": "CPU float64 eigvalsh of sense-normalized P",
            "min_eigenvalue": smallest, "negative_eigenvalue_tolerance": 0.0,
            "scope": "numerical validation, not an exact PSD certificate"}


def _dense(matrix, rows: int, cols: int) -> np.ndarray:
    if matrix.format_ != highspy.MatrixFormat.kColwise:
        raise ValueError(f"unsupported highspy matrix format: {matrix.format_}")
    starts, indices, values = matrix.start_, matrix.index_, matrix.value_
    if len(starts) != cols + 1 or starts[0] != 0 or starts[-1] != len(values) or len(indices) != len(values):
        raise ValueError("invalid reader sparse structure")
    result = np.zeros((rows, cols))
    for j in range(cols):
        previous = set()
        if not 0 <= starts[j] <= starts[j + 1] <= len(values):
            raise ValueError("invalid reader sparse pointers")
        for k in range(starts[j], starts[j + 1]):
            i = indices[k]
            if not 0 <= i < rows or i in previous:
                raise ValueError("invalid/duplicate reader sparse index")
            previous.add(i)
            result[i, j] = values[k]
    return result


def read_model(snapshot: Path, log: Path, max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES) -> dict:
    """Read a byte-frozen .mps snapshot; never run the solver or presolve."""
    audited = audit_records(snapshot.read_bytes(), max_dense_bytes)
    reader = highspy.Highs()
    for key, value in (("log_file", str(log)), ("log_to_console", False),
                       ("threads", 1), ("presolve", "off")):
        status = reader.setOptionValue(key, value)
        if status != highspy.HighsStatus.kOk:
            raise ValueError(f"highspy option {key}: {status}")
    status = reader.readModel(str(snapshot))
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"highspy readModel failed/warned: status={status} source={snapshot} log={log}")
    model = reader.getModel()
    lp, hessian = model.lp_, model.hessian_
    if any(kind != highspy.HighsVarType.kContinuous for kind in lp.integrality_):
        raise ValueError("integer/noncontinuous reader model rejected")
    n, m = lp.num_col_, lp.num_row_
    if lp.col_names_ != audited["variable_names"] or lp.row_names_ != audited["row_names"]:
        raise ValueError("reader coordinate names/order differ from raw records")
    if (n, m) != (len(audited["variable_names"]), len(audited["row_names"])):
        raise ValueError("reader dimensions differ from raw records")
    P = np.zeros((n, n))
    if hessian.dim_:
        if hessian.dim_ != n or hessian.format_ != highspy.HessianFormat.kTriangular:
            raise ValueError("unsupported reader Hessian dimensions/format")
        class Matrix:
            format_ = highspy.MatrixFormat.kColwise
            start_, index_, value_ = hessian.start_, hessian.index_, hessian.value_
        triangle = _dense(Matrix(), n, n)
        P = triangle + triangle.T - np.diag(np.diag(triangle))
    actual = dict(zip(ARRAY_NAMES, (
        P, np.asarray(lp.col_cost_, dtype="<f8"), _dense(lp.a_matrix_, m, n),
        np.asarray(lp.row_lower_, dtype="<f8"), np.asarray(lp.row_upper_, dtype="<f8"),
        np.asarray(lp.col_lower_, dtype="<f8"), np.asarray(lp.col_upper_, dtype="<f8"),
        np.asarray(lp.offset_, dtype="<f8"),
    )))
    sense = "minimize" if lp.sense_ == highspy.ObjSense.kMinimize else "maximize"
    if sense != audited["objective_sense"]:
        raise ValueError("reader objective sense differs from raw records")
    for name in ARRAY_NAMES:
        if not np.array_equal(actual[name], audited["arrays"][name]):
            raise ValueError(f"reader semantics differ from raw records: {name}")
    audited["arrays"] = actual
    audited["validation"] = validate_arrays(audited)
    audited["reader"] = {"package": "highspy", "version": importlib.metadata.version("highspy"),
                         "engine_version": reader.version(), "device": "cpu",
                         "presolve": "off", "threads": 1, "status": "kOk",
                         "raw_record_semantics": "exactly matched"}
    return audited


def array_hashes(arrays: dict) -> dict:
    result = {}
    for name in ARRAY_NAMES:
        value = np.asarray(arrays[name], dtype="<f8", order="C")
        descriptor = json.dumps({"name": name, "shape": list(value.shape), "dtype": "<f8"},
                                sort_keys=True, separators=(",", ":")).encode("ascii")
        result[name] = sha256(descriptor + b"\0" + value.tobytes(order="C"))
    return result


def model_hash(model: dict) -> str:
    semantic = {key: model[key] for key in ("variable_names", "row_names", "objective_row",
                                            "objective_sense", "model_name")}
    semantic["array_hashes"] = array_hashes(model["arrays"])
    return sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("ascii"))


def solver_arrays(model: dict) -> dict:
    """Original x coordinates; append identity bounds, normalize sense only.

    The core has no offset argument. Consumers must add returned offset and
    apply objective_sign to report values in the original objective sense.
    """
    validate_arrays(model)
    a = model["arrays"]
    n = len(a["q"])
    sign = 1 if model["objective_sense"] == "minimize" else -1
    return {"P": sign * a["P"], "q": sign * a["q"],
            "A": np.concatenate((a["A"], np.eye(n))),
            "l": np.concatenate((a["l"], a["variable_lower"])),
            "u": np.concatenate((a["u"], a["variable_upper"])),
            "offset": sign * float(a["offset"]), "objective_sign": sign,
            "row_mapping": [("row", i) for i in range(len(a["l"]))]
                           + [("variable_bound", i) for i in range(n)]}


def original_objective(model: dict, x: np.ndarray) -> float:
    a = model["arrays"]
    x = np.asarray(x, dtype=np.float64)
    if x.shape != a["q"].shape or not np.isfinite(x).all():
        raise ValueError("objective needs finite original-coordinate x")
    value = float(0.5 * x @ a["P"] @ x + a["q"] @ x + a["offset"])
    if not math.isfinite(value):
        raise ValueError("nonfinite original objective")
    return value


def freeze(manifest_path: Path, local_dir: Path, output: Path,
           max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES) -> Path:
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "importer_version": VERSION, "verdict": "RUNNING",
              "started_at": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
              "cases": [], "current_case": None,
              "source_hashes": {
                  "scripts/public_qp_corpus.py": sha256(Path(__file__).read_bytes()),
              }}
    try:
        raw_manifest = manifest_path.read_bytes()
        with (output / "input-manifest.json").open("xb") as stream:
            stream.write(raw_manifest)
        report["input_manifest_sha256"] = sha256(raw_manifest)
        entries = select_inputs(read_json(output / "input-manifest.json"), local_dir)
        for entry, source in entries:
            report["current_case"] = entry["case_id"]
            raw = source.read_bytes()
            if sha256(raw) != entry["input_sha256"]:
                raise ValueError(f"input SHA256 mismatch: case={entry['case_id']} path={source}")
            case_dir = output / entry["case_id"]
            case_dir.mkdir()
            snapshot = case_dir / "raw.mps"
            with snapshot.open("xb") as stream:
                stream.write(raw)
            model = read_model(snapshot, case_dir / "reader.log", max_dense_bytes)
            if snapshot.read_bytes() != raw:
                raise ValueError("snapshot changed during reader execution")
            archive = case_dir / "arrays.npz"
            with archive.open("xb") as stream:
                np.savez(stream, **model["arrays"])
            with np.load(archive, allow_pickle=False) as loaded:
                if loaded.files != list(ARRAY_NAMES):
                    raise ValueError("NPZ roundtrip array membership mismatch")
                for name in ARRAY_NAMES:
                    if (loaded[name].shape != model["arrays"][name].shape
                            or loaded[name].dtype != model["arrays"][name].dtype
                            or loaded[name].tobytes() != model["arrays"][name].tobytes()):
                        raise ValueError(f"NPZ roundtrip byte/shape/dtype mismatch: {name}")
            metadata = {key: value for key, value in model.items() if key != "arrays"}
            metadata.update({
                "schema_version": 1, "importer_version": VERSION, "provenance": entry,
                "source_local_path": str(source), "raw_sha256": sha256(raw),
                "array_hashes": array_hashes(model["arrays"]), "model_sha256": model_hash(model),
                "npz_sha256": sha256(archive.read_bytes()), "scope": SCOPE,
                "qualification": "UNVERIFIED" if entry["origin"] == "public" else "SYNTHETIC_NOT_COVERAGE",
            })
            write_json(case_dir / "metadata.json", metadata)
            report["cases"].append({
                "case_id": entry["case_id"], "origin": entry["origin"],
                "raw_sha256": metadata["raw_sha256"], "model_sha256": metadata["model_sha256"],
                "metadata_sha256": sha256((case_dir / "metadata.json").read_bytes()),
            })
        report["verdict"] = "FROZEN"
        report["case_count"] = len(report["cases"])
        report["public_declared_count"] = sum(e["origin"] == "public" for e in report["cases"])
        report["public_qualified_count"] = 0
        report["industrial_qualified_count"] = 0
        report["current_case"] = None
    except Exception as error:
        report["verdict"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        with (output / "failure.log").open("x", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "corpus.json", report)
    pinned = sha256((output / "corpus.json").read_bytes())
    verify_frozen(output, pinned)
    return output / "corpus.json"


def verify_frozen(output: Path, expected_manifest_sha256: str,
                  max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES) -> dict:
    """Read-only replay anchored in an externally supplied manifest hash."""
    output = output.expanduser().resolve(strict=True)
    _sha(expected_manifest_sha256, "expected_manifest_sha256")
    raw = (output / "corpus.json").read_bytes()
    if sha256(raw) != expected_manifest_sha256:
        raise ValueError("frozen corpus manifest SHA256 mismatch")
    report = read_json(output / "corpus.json")
    if (report.get("verdict") != "FROZEN" or report.get("importer_version") != VERSION
            or report.get("schema_version") != 1):
        raise ValueError("not a supported successful frozen corpus")
    hashes = report.get("source_hashes")
    if not isinstance(hashes, dict) or set(hashes) != {"scripts/public_qp_corpus.py"}:
        raise ValueError("missing importer source SHA256")
    _sha(hashes["scripts/public_qp_corpus.py"], "importer source SHA256")
    entries = report.get("cases")
    if not isinstance(entries, list) or not entries or report.get("case_count") != len(entries):
        raise ValueError("empty/incomplete frozen corpus rejected")
    input_raw = (output / "input-manifest.json").read_bytes()
    if sha256(input_raw) != report["input_manifest_sha256"]:
        raise ValueError("input manifest byte hash mismatch")
    input_manifest = read_json(output / "input-manifest.json")
    if (set(input_manifest) != {"schema_version", "cases"}
            or input_manifest["schema_version"] != 1
            or not isinstance(input_manifest["cases"], list) or not input_manifest["cases"]):
        raise ValueError("empty/unsupported original input manifest")
    declared = [declaration(e) for e in input_manifest["cases"]]
    if [e["case_id"] for e in entries] != [e["case_id"] for e in declared]:
        raise ValueError("frozen membership differs from input manifest")
    ids = set()
    for row, entry in zip(entries, declared):
        case_id = entry["case_id"]
        if case_id in ids:
            raise ValueError("duplicate frozen case")
        ids.add(case_id)
        case_dir = (output / case_id).resolve(strict=True)
        if not case_dir.is_relative_to(output):
            raise ValueError("frozen case escaped corpus")
        metadata_path = case_dir / "metadata.json"
        if sha256(metadata_path.read_bytes()) != row["metadata_sha256"]:
            raise ValueError(f"metadata SHA256 mismatch: {case_id}")
        metadata = read_json(metadata_path)
        if metadata["schema_version"] != 1 or metadata["importer_version"] != VERSION:
            raise ValueError("unsupported frozen metadata schema")
        if metadata["provenance"] != entry:
            raise ValueError("frozen provenance differs from input declaration")
        raw = (case_dir / "raw.mps").read_bytes()
        if sha256(raw) != entry["input_sha256"] or sha256(raw) != metadata["raw_sha256"] or sha256(raw) != row["raw_sha256"]:
            raise ValueError(f"raw input SHA256 mismatch: {case_id}")
        archive = case_dir / "arrays.npz"
        if sha256(archive.read_bytes()) != metadata["npz_sha256"]:
            raise ValueError(f"NPZ byte SHA256 mismatch: {case_id}")
        audited = audit_records(raw, max_dense_bytes)
        with np.load(archive, allow_pickle=False) as loaded:
            if len(loaded.files) != len(ARRAY_NAMES) or set(loaded.files) != set(ARRAY_NAMES):
                raise ValueError("frozen array membership mismatch")
            model = {**metadata, "arrays": {name: loaded[name] for name in ARRAY_NAMES}}
        validation = validate_arrays(model)
        for key in ("variable_names", "row_names", "objective_row", "objective_sense", "model_name"):
            if model[key] != audited[key]:
                raise ValueError(f"frozen coordinate semantics mismatch: {key}")
        for name in ARRAY_NAMES:
            if not np.array_equal(model["arrays"][name], audited["arrays"][name]):
                raise ValueError(f"frozen/raw semantics mismatch: {name}")
        if array_hashes(model["arrays"]) != metadata["array_hashes"]:
            raise ValueError("canonical frozen array SHA256 mismatch")
        if model_hash(model) != metadata["model_sha256"] or model_hash(model) != row["model_sha256"]:
            raise ValueError("frozen model SHA256 mismatch")
        if metadata["validation"] != validation or metadata["reader"]["raw_record_semantics"] != "exactly matched":
            raise ValueError("frozen validation metadata mismatch")
        expected_qualification = "UNVERIFIED" if entry["origin"] == "public" else "SYNTHETIC_NOT_COVERAGE"
        if metadata["qualification"] != expected_qualification or row["origin"] != entry["origin"]:
            raise ValueError("frozen qualification/origin mismatch")
    public = sum(e["origin"] == "public" for e in declared)
    if (report["public_declared_count"] != public or report["public_qualified_count"] != 0
            or report["industrial_qualified_count"] != 0):
        raise ValueError("unsupported public/industrial qualification claim")
    return {"verdict": "VERIFIED", "checked_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(entries), "corpus_sha256": expected_manifest_sha256,
            "public_declared_count": public, "public_qualified_count": 0,
            "industrial_qualified_count": 0}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import")
    importer.add_argument("--manifest", type=Path, required=True)
    importer.add_argument("--local-dir", type=Path, required=True)
    importer.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("verify")
    replay.add_argument("--corpus", type=Path, required=True)
    replay.add_argument("--manifest-sha256", required=True)
    for command in (importer, replay):
        command.add_argument("--max-dense-bytes", type=int, default=DEFAULT_MAX_DENSE_BYTES)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "import":
            path = freeze(args.manifest, args.local_dir, args.artifact_dir, args.max_dense_bytes)
            result = {"verdict": "FROZEN", "manifest": str(path),
                      "manifest_sha256": sha256(path.read_bytes())}
        else:
            result = verify_frozen(args.corpus, args.manifest_sha256, args.max_dense_bytes)
        print("PUBLIC_QP_CORPUS " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

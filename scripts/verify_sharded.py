"""Plan, run and independently recheck source-pinned full CPU regression.

Pytest supplies the complete collection, not guessed test globs. Historical
LPT assigns whole files by default or explicitly audited fixture components.
Each worker executes the existing verify entry. Artifacts are external,
timeouts fail, and no speedup is inferred.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/sharded-regression")
MAX_CPU_CAPACITY = 32
FIXTURE_CONTRACT = "fixture-components-v2"
ISOLATED_BUILTINS = {
    "tmp_path_factory": {
        "module": "_pytest.tmpdir", "scope": "session", "pytest_version": "8.4.2",
        "sha256": "236918c090160c1f6b935e162ff44ab273a700221d5f40ac15892be75e4503ee",
        "policy": "unique-mktemp-and-worker-basetemp-v1",
    },
    "pytestconfig": {
        "module": "_pytest.fixtures", "scope": "session", "pytest_version": "8.4.2",
        "sha256": "41dfd72c1e6da619906bf84945694c1cb9b919bad18303278fac35630c8c7748",
        "policy": "config-read-assertions-only-v1",
    },
}


class GateError(RuntimeError):
    """A preserved, named admission or acceptance failure."""


def require(condition, message):
    if not condition:
        raise GateError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_bytes(encoded(value))


def read(path, pin=None):
    if pin is not None:
        require(digest(path) == pin, f"hash_mismatch: {path}")
    return json.loads(Path(path).read_bytes())


def external(directory, repo):
    directory = Path(directory).expanduser().resolve()
    require(not directory.is_relative_to(repo.resolve()), "artifacts_inside_repository")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def new_artifacts(directory, repo, prefix):
    return Path(tempfile.mkdtemp(prefix=prefix, dir=external(directory, repo)))


def verifier(repo):
    path = repo / "scripts/verify.py"
    require(path.is_file(), "missing_verify_entry")
    spec = importlib.util.spec_from_file_location("_openopt_verify_sharded_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = repo
    return module


def snapshot(repo):
    value = verifier(repo).source_snapshot(repo)
    require(value["clean"], "source_not_clean")
    return value


def consistent(repo, expected):
    actual = snapshot(repo)
    result = verifier(repo).check_source_consistency(expected, actual)
    require(result["verdict"] == "PASS", f"source_drift: {result}")
    return result


def cpu_env(repo, artifacts):
    env = os.environ.copy()
    # An explicit CPU regression never consumes a local GPU.
    env.update({
        "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "",
        "PYTEST_PLUGINS": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "TMPDIR": str(artifacts),
        "TRITON_CACHE_DIR": str(artifacts / "triton-cache"),
        "PYTHONPATH": os.pathsep.join(
            [str(repo), str(repo / "src"), str(repo / "benchmarks")]),
    })
    return env


def deadline(value):
    require(math.isfinite(value) and 0 < value <= 300, "timeout_must_be_in_0_300")
    return value


def capacity(value, name):
    require(type(value) is int and 1 <= value <= MAX_CPU_CAPACITY,
            f"{name}_must_be_1_{MAX_CPU_CAPACITY}")
    return value


def command_logged(command, repo, artifacts, timeout, env):
    deadline(timeout)
    started = time.monotonic()
    metadata = {
        "command": command, "started_at": now(), "returncode": None,
        "timeout": False, "timeout_s": timeout,
    }
    with (artifacts / "process.log").open("x", encoding="utf-8") as log:
        log.write(f"command={command!r}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        metadata["pid"] = process.pid
        try:
            metadata["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            metadata["timeout"] = True
            # This process group contains only this tool's admitted worker.
            os.killpg(process.pid, signal.SIGKILL)
            metadata["returncode"] = process.wait()
        finally:
            metadata.update(finished_at=now(), wall_s=time.monotonic() - started)
            log.write(f"\nreturncode={metadata['returncode']} timeout={metadata['timeout']}\n")
            save(artifacts / "process.json", metadata)
    require(not metadata["timeout"], f"worker_timeout: {artifacts}")
    require(metadata["returncode"] == 0,
            f"process_failed_rc={metadata['returncode']}: {artifacts}")
    return metadata


def junit_key(nodeid):
    from _pytest.junitxml import mangle_test_address
    parts = mangle_test_address(nodeid)
    return ".".join(parts[:-1]), parts[-1]


def pytest_collection_finish(session):
    """Explicit collection plugin, not a substitute for pytest discovery."""
    output = os.environ.get("OPENOPT_SHARD_COLLECTION")
    if output:
        result = {"nodeids": [item.nodeid for item in session.items]}
        if os.environ.get("OPENOPT_SHARD_GROUPING") == FIXTURE_CONTRACT:
            # Persist diagnostics even if pytest reports this hook's failure.
            try:
                selected = json.loads(os.environ.get("OPENOPT_SHARD_FIXTURE_FILES", "[]"))
                result["fixture_audit"] = audit_fixtures(session, selected)
            except Exception as error:
                result["fixture_error"] = f"{type(error).__name__}: {error}"
                save(Path(output), result)
                raise
        save(Path(output), result)


def collect(repo, artifacts, timeout=60, grouping="file", fixture_files=()):
    env = cpu_env(repo, artifacts)
    env["OPENOPT_SHARD_COLLECTION"] = str(artifacts / "collection.json")
    env["OPENOPT_SHARD_GROUPING"] = grouping
    env["OPENOPT_SHARD_FIXTURE_FILES"] = json.dumps(list(fixture_files))
    command_logged(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "scripts.verify_sharded", "-o", "addopts=",
         "-o", f"cache_dir={artifacts / 'pytest-cache'}",
         "--basetemp", str(artifacts / "pytest-tmp")],
        repo, artifacts, timeout, env,
    )
    nodes = read(artifacts / "collection.json")["nodeids"]
    require(isinstance(nodes, list) and nodes, "empty_collection")
    require(all(isinstance(node, str) and "::" in node for node in nodes),
            "invalid_collection_nodeid")
    require(len(nodes) == len(set(nodes)), "duplicate_collection_nodeid")
    keys = [junit_key(node) for node in nodes]
    require(len(keys) == len(set(keys)), "ambiguous_junit_identity")
    for node in nodes:
        target = Path(node.split("::", 1)[0])
        require(not target.is_absolute() and ".." not in target.parts
                and (repo / target).is_file(), f"invalid_collection_path: {node}")
    return nodes


def static_fixture_file(path):
    """Restricted source audit; runtime closure must agree with declarations."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    pytest_names, fixture_names = set(), set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            pytest_names.update(alias.asname or alias.name for alias in statement.names
                                if alias.name == "pytest")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "pytest":
            fixture_names.update(alias.asname or alias.name for alias in statement.names
                                 if alias.name == "fixture")

    def is_fixture(expression):
        return (isinstance(expression, ast.Name) and expression.id in fixture_names
                or isinstance(expression, ast.Attribute) and expression.attr == "fixture"
                and isinstance(expression.value, ast.Name)
                and expression.value.id in pytest_names)

    declarations = {}
    request_parameters = {id(arg) for entry in ast.walk(tree)
                          if isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef))
                          for arg in entry.args.args + entry.args.posonlyargs
                          if arg.arg == "request"}
    for entry in ast.walk(tree):
        line = getattr(entry, "lineno", 0)
        require(not (isinstance(entry, ast.Attribute) and entry.attr == "getfixturevalue"),
                f"dynamic_fixture_lookup: {path}:{line}")
        require(not (isinstance(entry, ast.Name)
                     and entry.id in {"pytest_plugins", "exec", "eval", "__import__"}),
                f"dynamic_fixture_environment: {path}:{line}")
        require(not (isinstance(entry, ast.Attribute)
                     and isinstance(entry.ctx, (ast.Store, ast.Del))
                     and isinstance(entry.value, ast.Name)
                     and entry.value.id in pytest_names),
                f"dynamic_fixture_environment: {path}:{line}")
        if isinstance(entry, ast.arg) and entry.arg == "request":
            require(id(entry) in request_parameters, f"unmodelled_fixture_request: {path}:{line}")
        if isinstance(entry, ast.Name) and entry.id == "request":
            require(any(isinstance(owner, ast.Attribute) and owner.value is entry
                        and owner.attr == "param" for owner in ast.walk(tree)),
                    f"unmodelled_fixture_request: {path}:{entry.lineno}")
        require(not (isinstance(entry, ast.Call) and isinstance(entry.func, ast.Name)
                     and entry.func.id in {"getattr", "setattr", "delattr", "globals", "locals"}
                     and any(isinstance(arg, ast.Name) and arg.id in {"pytest", "request"}
                             for arg in entry.args)),
                f"dynamic_fixture_reflection: {path}:{line}")
        if isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef)):
            require(not entry.name.startswith("pytest_"),
                    f"dynamic_pytest_hook: {path}:{entry.lineno}")
            require(entry.name not in {"setup_module", "teardown_module", "setUpModule",
                                       "tearDownModule", "setup_class", "teardown_class",
                                       "setUpClass", "tearDownClass"},
                    f"unmodelled_xunit_scope: {path}:{entry.lineno}")
            for decorator in entry.decorator_list:
                expression = decorator.func if isinstance(decorator, ast.Call) else decorator
                if not is_fixture(expression):
                    continue
                scope, name = "function", entry.name
                if isinstance(decorator, ast.Call):
                    require(not decorator.args, f"dynamic_fixture_decorator: {path}:{entry.lineno}")
                    for keyword in decorator.keywords:
                        require(keyword.arg in {"scope", "name", "autouse", "params", "ids"},
                                f"dynamic_fixture_decorator: {path}:{entry.lineno}")
                        if keyword.arg in {"scope", "name"}:
                            require(isinstance(keyword.value, ast.Constant)
                                    and isinstance(keyword.value.value, str),
                                    f"dynamic_fixture_scope: {path}:{entry.lineno}")
                            if keyword.arg == "scope":
                                scope = keyword.value.value
                            else:
                                name = keyword.value.value
                        if keyword.arg == "autouse":
                            require(isinstance(keyword.value, ast.Constant)
                                    and type(keyword.value.value) is bool,
                                    f"dynamic_fixture_autouse: {path}:{entry.lineno}")
                require(scope in {"function", "class", "module", "session"},
                        f"unsupported_fixture_scope: {path}:{entry.lineno}")
                require(len(entry.decorator_list) == 1,
                        f"wrapped_fixture: {path}:{entry.lineno}")
                parameters = [arg.arg for arg in entry.args.posonlyargs + entry.args.args]
                require(not entry.args.vararg and not entry.args.kwarg
                        and not entry.args.kwonlyargs and not entry.args.defaults,
                        f"dynamic_fixture_signature: {path}:{entry.lineno}")
                if parameters[:1] in (["self"], ["cls"]):
                    parameters = parameters[1:]
                key = (entry.name, entry.lineno)
                declarations[key] = {"name": name, "scope": scope, "dependencies": parameters,
                                     "first_line": min(entry.lineno, decorator.lineno)}
    return tree, declarations


def builtin_isolation(name, fixture, path, version, pytest_path):
    policy = ISOLATED_BUILTINS.get(name)
    if policy is None:
        return None
    origin = Path(pytest_path).resolve().parent.parent / (
        policy["module"].replace(".", "/") + ".py")
    require(fixture.func.__module__ == policy["module"]
            and fixture.scope == policy["scope"]
            and fixture.func.__name__ == name and Path(path).resolve() == origin
            and version == policy["pytest_version"] and digest(path) == policy["sha256"],
            f"builtin_isolation_not_qualified: {name}")
    return {**policy, "origin": str(origin)}


def audit_builtin_usage(tree, name, path):
    """Only direct qualified reads/unique allocation may omit a session edge."""
    parents = {child: entry for entry in ast.walk(tree) for child in ast.iter_child_nodes(entry)}
    for entry in ast.walk(tree):
        if not isinstance(entry, ast.Name) or entry.id != name:
            continue
        attribute = parents.get(entry)
        call = parents.get(attribute)
        require(isinstance(attribute, ast.Attribute) and attribute.value is entry
                and isinstance(call, ast.Call) and call.func is attribute,
                f"builtin_isolation_usage_unproven: {path}:{entry.lineno}:{name}")
        if name == "tmp_path_factory":
            require(attribute.attr == "mktemp" and len(call.args) == 1
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                    and all(keyword.arg == "numbered" and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True for keyword in call.keywords),
                    f"builtin_temp_not_unique: {path}:{entry.lineno}")
        else:
            require(attribute.attr in {"getoption", "getini", "get_verbosity"}
                    and all(isinstance(arg, ast.Constant) for arg in call.args)
                    and all(keyword.arg is not None and isinstance(keyword.value, ast.Constant)
                            for keyword in call.keywords),
                    f"builtin_config_not_readonly: {path}:{entry.lineno}")
            comparison = parents.get(call)
            assertion = parents.get(comparison)
            require(isinstance(comparison, ast.Compare)
                    and len(comparison.ops) == 1
                    and isinstance(comparison.ops[0], (ast.Eq, ast.NotEq))
                    and isinstance(assertion, ast.Assert) and assertion.test is comparison,
                    f"builtin_config_result_may_escape: {path}:{entry.lineno}")
            other = (comparison.comparators[0] if comparison.left is call
                     else comparison.left)
            require(isinstance(other, ast.Constant)
                    or isinstance(other, (ast.List, ast.Tuple))
                    and all(isinstance(value, ast.Constant) for value in other.elts),
                    f"builtin_config_result_may_escape: {path}:{entry.lineno}")


def audit_fixtures(session, selected):
    """Cross-check AST with pytest's actual closure, then union scoped resources.

    This deliberately depends on pytest's private closure metadata. Missing or
    changed metadata rejects opt-in admission rather than guessing dependencies.
    """
    import pytest

    repo = Path(session.config.rootpath).resolve()
    files, parsed, runtime, isolated = {}, {}, {}, {}
    require(isinstance(selected, list) and selected
            and all(isinstance(file, str) and file for file in selected)
            and selected == sorted(set(selected)), "explicit_fixture_files_required")
    collected_files = {item.nodeid.split("::", 1)[0] for item in session.items}
    require(set(selected).issubset(collected_files), "fixture_file_not_in_full_collection")

    def audited_file(path):
        path = Path(path).resolve()
        require(path.is_relative_to(repo), f"fixture_source_outside_repo: {path}")
        relative = str(path.relative_to(repo))
        if relative not in parsed:
            parsed[relative] = static_fixture_file(path)
            files[relative] = digest(path)
        return relative, parsed[relative]

    for plugin_name, plugin in session.config.pluginmanager.list_name_plugin():
        if plugin is None:
            continue
        module = (plugin.__name__ if inspect.ismodule(plugin)
                  else getattr(plugin, "__module__", type(plugin).__module__))
        path = getattr(plugin, "__file__", None)
        if path and Path(path).name == "conftest.py":
            audited_file(path)
        else:
            require(module.startswith("_pytest.") or module in {"_pytest", "pytest",
                    "scripts.verify_sharded"},
                    f"unknown_fixture_plugin: {plugin_name}")

    builtin = {
        "monkeypatch": ("_pytest.monkeypatch", "function"),
        "tmp_path": ("_pytest.tmpdir", "function"),
        "tmp_path_factory": ("_pytest.tmpdir", "session"),
        "pytestconfig": ("_pytest.fixtures", "session"),
        "capsys": ("_pytest.capture", "function"),
        "capfd": ("_pytest.capture", "function"),
        "capsysbinary": ("_pytest.capture", "function"),
        "capfdbinary": ("_pytest.capture", "function"),
        "caplog": ("_pytest.logging", "function"),
        "recwarn": ("_pytest.recwarn", "function"),
    }
    rows = []
    session_dependencies = {}
    for item in session.items:
        relative = item.nodeid.split("::", 1)[0]
        if relative not in selected:
            rows.append({"nodeid": item.nodeid, "constraints": [f"wholefile:{relative}"],
                         "closure": [], "audited": False})
            continue
        require(type(item).__module__ == "_pytest.python"
                and type(item).__name__ == "Function", f"unknown_pytest_item: {item.nodeid}")
        relative, (tree, _) = audited_file(item.path)
        info = getattr(item, "_fixtureinfo", None)
        require(info is not None, f"missing_fixture_closure: {item.nodeid}")
        function = inspect.unwrap(item.obj)
        require(getattr(item.obj, "__code__", None) == function.__code__,
                f"wrapped_test_callable: {item.nodeid}")
        require(Path(inspect.getsourcefile(function)).resolve() == Path(item.path).resolve(),
                f"imported_or_wrapped_test: {item.nodeid}")
        definition = next((entry for entry in ast.walk(tree)
                           if isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and entry.name == function.__name__
                           and min([entry.lineno] + [d.lineno for d in entry.decorator_list])
                           == function.__code__.co_firstlineno), None)
        require(definition is not None, f"unresolved_test_ast: {item.nodeid}")
        for decorator in definition.decorator_list:
            expression = decorator.func if isinstance(decorator, ast.Call) else decorator
            require(isinstance(expression, ast.Attribute)
                    and isinstance(expression.value, ast.Attribute)
                    and expression.value.attr == "mark"
                    and isinstance(expression.value.value, ast.Name)
                    and expression.value.value.id == "pytest"
                    and expression.attr in {"parametrize", "skip", "skipif", "xfail",
                                            "usefixtures", "filterwarnings"},
                    f"unknown_test_decorator: {item.nodeid}")
        class_nodes = [node for node in item.listchain() if type(node).__name__ == "Class"]
        constraints = []
        if class_nodes:
            require(len(class_nodes) == 1, f"nested_test_class: {item.nodeid}")
            class_node = class_nodes[0]
            class_ast = next((entry for entry in tree.body if isinstance(entry, ast.ClassDef)
                              and entry.name == class_node.name), None)
            require(class_ast is not None and not class_ast.bases and not class_ast.decorator_list,
                    f"dynamic_test_class: {item.nodeid}")
            constraints.append(f"class:{class_node.nodeid}")
        callspec = getattr(item, "callspec", None)
        direct = set()
        if callspec is not None:
            for name, scope in callspec._arg2scope.items():
                require(scope.value == "function", f"nonfunction_param_scope: {item.nodeid}:{name}")
                if name not in info.name2fixturedefs:
                    direct.add(name)
        closure = []
        for name in info.names_closure:
            if name == "request" or name in direct:
                continue
            definitions = info.name2fixturedefs.get(name)
            require(definitions and len(definitions) == 1,
                    f"unresolved_or_overridden_fixture: {item.nodeid}:{name}")
            fixture = definitions[-1]
            func = inspect.unwrap(fixture.func)
            if (func.__module__ == "_pytest.python"
                    and func.__name__ == "get_direct_param_fixture_func"):
                require(callspec is not None and name in callspec.params
                        and fixture.scope == "function",
                        f"unknown_direct_parameter: {item.nodeid}:{name}")
                continue
            module = func.__module__
            path = Path(inspect.getsourcefile(func)).resolve()
            scope = fixture.scope
            isolation = None
            if module.startswith("_pytest."):
                require(builtin.get(name) == (module, scope),
                        f"unknown_builtin_fixture: {item.nodeid}:{name}")
                runtime[str(path)] = digest(path)
                identity = f"builtin:{module}:{name}:{scope}"
                isolation = builtin_isolation(name, fixture, path, pytest.__version__, pytest.__file__)
                if isolation is not None:
                    isolated[name] = isolation
            else:
                source, (_, declared) = audited_file(path)
                statement = next((entry for (func_name, _), entry in declared.items()
                                  if func_name == func.__name__
                                  and entry["name"] == name
                                  and entry["first_line"] == func.__code__.co_firstlineno), None)
                require(statement is not None and statement["scope"] == scope
                        and statement["dependencies"] == list(fixture.argnames),
                        f"fixture_ast_runtime_conflict: {item.nodeid}:{name}")
                identity = f"{source}:{func.__qualname__}:{name}:{scope}"
            closure.append({"identity": identity, "scope": scope,
                            "dependencies": list(fixture.argnames),
                            "isolated_builtin": isolation})
            require(scope in {"function", "class", "module", "session"},
                    f"unsupported_fixture_scope: {item.nodeid}:{name}")
            if scope != "function" and isolation is None:
                owner = ("session" if scope == "session" else relative if scope == "module"
                         else class_nodes[0].nodeid if class_nodes else item.nodeid)
                constraints.append(f"fixture:{identity}:owner:{owner}")
                if scope == "session":
                    session_dependencies[fixture] = identity
        rows.append({"nodeid": item.nodeid, "constraints": sorted(set(constraints)),
                     "closure": closure, "audited": True})
    for source, (tree, _) in parsed.items():
        for name in isolated:
            audit_builtin_usage(tree, name, source)
    # Do not audit unrelated whole files or guess their dynamic dependencies.
    # A visible session edge crossing the selected boundary needs explicit audit.
    for item in session.items:
        if item.nodeid.split("::", 1)[0] in selected:
            continue
        info = getattr(item, "_fixtureinfo", None)
        require(info is not None or not session_dependencies,
                f"unknown_external_session_closure: {item.nodeid}")
        if info is None:
            continue
        for definitions in info.name2fixturedefs.values():
            for fixture, identity in session_dependencies.items():
                require(fixture not in definitions,
                        f"session_dependency_outside_fixture_selection: {identity}; "
                        f"dependent={item.nodeid}; explicitly_audit_dependent_file")
    return {"contract": FIXTURE_CONTRACT, "pytest_version": pytest.__version__,
            "selected_files": selected, "source_sha256": files,
            "runtime_sha256": runtime, "isolated_builtins": isolated, "nodes": rows}


def fixture_groups(nodes, audit):
    require(audit.get("contract") == FIXTURE_CONTRACT, "invalid_fixture_contract")
    rows = audit.get("nodes")
    require(isinstance(rows, list) and [row["nodeid"] for row in rows] == nodes,
            "fixture_collection_mismatch")
    parent = {node: node for node in nodes}

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owners = {}
    for row in rows:
        require(isinstance(row["constraints"], list)
                and all(isinstance(value, str) and value for value in row["constraints"])
                and row["constraints"] == sorted(set(row["constraints"])),
                "invalid_fixture_constraints")
        for constraint in row["constraints"]:
            if constraint in owners:
                parent[root(row["nodeid"])] = root(owners[constraint])
            else:
                owners[constraint] = row["nodeid"]
    groups = {}
    for row in rows:
        group = groups.setdefault(root(row["nodeid"]), {"nodeids": [], "constraints": set()})
        group["nodeids"].append(row["nodeid"])
        group["constraints"].update(row["constraints"])
    return [{"id": index, "nodeids": group["nodeids"],
             "constraints": sorted(group["constraints"])}
            for index, group in enumerate(sorted(groups.values(), key=lambda g: g["nodeids"][0]))]


def parse_junit(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    require(bool(suites), "missing_junit_testsuite")
    records = []
    for suite in suites:
        cases = suite.findall("testcase")
        require(int(suite.get("tests", "-1")) == len(cases), "junit_count_mismatch")
        for field, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            require(int(suite.get(field, "0")) ==
                    sum(case.find(tag) is not None for case in cases),
                    f"junit_{field}_mismatch")
        for case in cases:
            seconds = float(case.get("time", "nan"))
            require(math.isfinite(seconds) and seconds >= 0, "invalid_junit_time")
            key = (case.get("classname"), case.get("name"))
            require(all(isinstance(part, str) and part for part in key), "missing_junit_identity")
            records.append({
                "key": key, "time_s": seconds,
                "failures": int(case.find("failure") is not None),
                "errors": int(case.find("error") is not None),
                "skipped": int(case.find("skipped") is not None),
            })
    keys = [record["key"] for record in records]
    require(len(keys) == len(set(keys)), "duplicate_junit_identity")
    require(bool(records), "empty_junit")
    counts = {"tests": len(records)}
    counts.update({field: sum(record[field] for record in records)
                   for field in ("failures", "errors", "skipped")})
    return records, counts


def lpt(nodes, histories, shards=3, unknown_s=1.0, components=None):
    capacity(shards, "shards")
    require(math.isfinite(unknown_s) and unknown_s > 0, "invalid_unknown_prediction")
    costs, provenance = {}, []
    for path in histories:
        records, counts = parse_junit(path)
        provenance.append({"path": str(Path(path).resolve()), "sha256": digest(path),
                           "tests": counts, "precedence": len(provenance)})
        for record in records:
            if not record["failures"] and not record["errors"]:
                costs[record["key"]] = record["time_s"]
    groups = {}
    component_by_node = {}
    if components is not None:
        for group in components:
            for node in group["nodeids"]:
                require(node not in component_by_node, "duplicate_fixture_group_node")
                component_by_node[node] = group["id"]
        require(set(component_by_node) == set(nodes), "fixture_group_membership_mismatch")
    unknown = []
    for node in nodes:
        key = junit_key(node)
        if key not in costs:
            unknown.append(node)
        target = (node.split("::", 1)[0] if components is None else component_by_node[node])
        group = groups.setdefault(target, {
            "nodeids": [], "predicted_test_s": 0.0, "unknown_nodeids": [],
        })
        group["nodeids"].append(node)
        group["predicted_test_s"] += costs.get(key, unknown_s)
        if key not in costs:
            group["unknown_nodeids"].append(node)
    require(len(groups) >= shards, "too_few_files_for_nonempty_shards" if components is None
            else "too_few_fixture_groups_for_nonempty_shards")
    assignments = [{"id": index, "targets": [], "nodeids": [], "predicted_test_s": 0.0,
                    "unknown_nodeids": [], "file_test_s": {}} for index in range(shards)]
    if components is not None:
        for assignment in assignments:
            assignment["group_test_s"] = {}
            assignment["group_ids"] = []
    for target, group in sorted(groups.items(),
                                key=lambda item: (-item[1]["predicted_test_s"], item[0])):
        chosen = min(assignments, key=lambda item: (
            item["predicted_test_s"], len(item["targets"]), item["id"]))
        chosen["targets"].extend([target] if components is None else group["nodeids"])
        chosen["nodeids"].extend(group["nodeids"])
        chosen["unknown_nodeids"].extend(group["unknown_nodeids"])
        if components is None:
            chosen["file_test_s"][target] = group["predicted_test_s"]
        else:
            chosen["group_ids"].append(target)
            chosen["group_test_s"][str(target)] = group["predicted_test_s"]
            for node in group["nodeids"]:
                file = node.split("::", 1)[0]
                chosen["file_test_s"][file] = (
                    chosen["file_test_s"].get(file, 0.0) + costs.get(junit_key(node), unknown_s))
        chosen["predicted_test_s"] += group["predicted_test_s"]
    if components is not None:
        for chosen in assignments:
            selected = set(chosen["nodeids"])
            files = sorted({node.split("::", 1)[0] for node in chosen["nodeids"]})
            chosen["targets"] = []
            for file in files:
                full = [node for node in nodes if node.split("::", 1)[0] == file]
                chosen["targets"].extend([file] if set(full).issubset(selected)
                                         else [node for node in full if node in selected])
    return assignments, provenance, unknown


def prediction_budget(assignments, timeout):
    deadline(timeout)
    for item in assignments:
        predictions = item.get("group_test_s", item["file_test_s"])
        for target, seconds in predictions.items():
            require(math.isfinite(seconds) and seconds >= 0, "invalid_file_prediction")
            kind = "fixture_group" if "group_test_s" in item else "file"
            require(seconds <= timeout,
                    f"{kind}_prediction_exceeds_deadline: {kind}={target} "
                    f"predicted_test_s={seconds:.6f} deadline_s={timeout:g}; "
                    f"{kind}_is_indivisible")
    for item in assignments:
        seconds = item["predicted_test_s"]
        require(math.isfinite(seconds) and seconds >= 0, "invalid_shard_prediction")
        require(seconds <= timeout,
                f"shard_prediction_exceeds_deadline: shard={item['id']} "
                f"predicted_test_s={seconds:.6f} deadline_s={timeout:g}; "
                "increase_cpu_shards_within_32_and_replan")


def validate_plan(plan):
    version = plan.get("schema_version")
    require(version in (1, 2) and plan.get("verdict") == "PASS",
            "invalid_plan_status")
    if version == 2:
        require(plan.get("grouping", {}).get("contract") == FIXTURE_CONTRACT
                and plan["grouping"].get("opt_in") is True,
                "fixture_grouping_opt_in_required")
    else:
        require("grouping" not in plan, "v1_cannot_claim_fixture_grouping")
    require(plan.get("prediction", {}).get("method") ==
            ("fixture-component-LPT" if version == 2 else "file-LPT"),
            "prediction_contract_mismatch")
    require(plan.get("coverage") == "tests", "invalid_plan_coverage")
    require(plan.get("source", {}).get("clean") is True, "unclean_plan_source")
    nodes = plan.get("collection", {}).get("nodeids")
    require(isinstance(nodes, list) and nodes and len(nodes) == len(set(nodes)),
            "empty_or_duplicate_plan_collection")
    require(all(isinstance(node, str) and "::" in node for node in nodes),
            "invalid_plan_nodeid")
    require(len({junit_key(node) for node in nodes}) == len(nodes), "ambiguous_junit_identity")
    assignments = plan.get("shards")
    require(isinstance(assignments, list) and 1 <= len(assignments) <= MAX_CPU_CAPACITY,
            "invalid_shards")
    require([item["id"] for item in assignments] == list(range(len(assignments))),
            "invalid_shard_ids")
    assigned_nodes, assigned_files = [], []
    for item in assignments:
        require(item["targets"] and item["nodeids"], "empty_shard_selection")
        if version == 1:
            require(set(item["targets"]) ==
                    {node.split("::", 1)[0] for node in item["nodeids"]}, "target_nodeid_conflict")
        else:
            expected_nodes = []
            for target in item["targets"]:
                require(isinstance(target, str), "invalid_fixture_target")
                expected_nodes.extend(node for node in nodes if node == target
                                      or node.split("::", 1)[0] == target)
            require(Counter(expected_nodes) == Counter(item["nodeids"]),
                    "target_nodeid_conflict")
        assigned_nodes.extend(item["nodeids"])
        assigned_files.extend(item["targets"])
    require(Counter(assigned_nodes) == Counter(nodes), "shard_collection_mismatch")
    require(len(assigned_files) == len(set(assigned_files)), "duplicate_shard_file")
    deadline(plan["shard_timeout_s"])
    history = plan["history"]
    require(isinstance(history, list), "invalid_prediction_history")
    for observation in history:
        read_path = Path(observation["path"])
        require(digest(read_path) == observation["sha256"], "prediction_history_pin_mismatch")
    components = None
    if version == 2:
        audit = plan["grouping"]["audit"]
        require(plan["grouping"].get("selected_files") == audit.get("selected_files")
                and isinstance(audit.get("selected_files"), list)
                and audit["selected_files"], "fixture_selection_contract_mismatch")
        collection = read(plan["collection"]["path"], plan["collection"]["sha256"])
        require(collection.get("fixture_audit") == audit
                and collection["nodeids"] == nodes, "fixture_sidecar_mismatch")
        for file, pin in audit["source_sha256"].items():
            require(plan["source"]["sha256"].get(file) == pin, "fixture_source_pin_mismatch")
        for file, pin in audit["runtime_sha256"].items():
            require(digest(file) == pin, "fixture_runtime_pin_mismatch")
        components = fixture_groups(nodes, audit)
        require(plan["grouping"]["groups"] == components, "fixture_group_recompute_mismatch")
    expected, provenance, unknown = lpt(
        nodes, [item["path"] for item in history], len(assignments),
        plan["prediction"]["unknown_test_s"], components,
    )
    require(provenance == history and unknown == plan["unknown_nodeids"],
            "prediction_provenance_mismatch")
    prediction_budget(expected, plan["shard_timeout_s"])
    require(assignments == expected, "prediction_assignment_mismatch")


def plan_run(repo, artifact_root, histories, shards=3, unknown_s=1.0,
             shard_timeout=300, collection_timeout=60, fixture_aware=False,
             fixture_files=()):
    artifacts = new_artifacts(artifact_root, repo, "plan-")
    require(type(fixture_aware) is bool, "fixture_opt_in_must_be_boolean")
    report = {"schema_version": 2 if fixture_aware else 1,
              "verdict": "RUNNING", "started_at": now()}
    try:
        require(bool(fixture_files) == fixture_aware, "explicit_fixture_files_required")
        require(len(fixture_files) == len(set(fixture_files)), "duplicate_fixture_file_selection")
        selected = sorted(fixture_files)
        capacity(shards, "shards")
        deadline(shard_timeout)
        report["source"] = snapshot(repo)
        directory = artifacts / "collection"
        directory.mkdir()
        nodes = collect(repo, directory, collection_timeout,
                        FIXTURE_CONTRACT if fixture_aware else "file", selected)
        report["collection"] = {"nodeids": nodes,
            "path": str(directory / "collection.json"),
            "sha256": digest(directory / "collection.json")}
        components = None
        if fixture_aware:
            audit = read(directory / "collection.json")["fixture_audit"]
            components = fixture_groups(nodes, audit)
            report["grouping"] = {"contract": FIXTURE_CONTRACT, "opt_in": True,
                                  "selected_files": selected, "audit": audit, "groups": components}
        assignments, historical, unknown = lpt(nodes, histories, shards, unknown_s, components)
        report.update({
            "repo": str(repo), "coverage": "tests", "collection": {"nodeids": nodes,
                "path": str(directory / "collection.json"),
                "sha256": digest(directory / "collection.json")},
            "shards": assignments, "history": historical, "unknown_nodeids": unknown,
            "prediction": {"method": "fixture-component-LPT" if fixture_aware else "file-LPT",
                "unknown_test_s": unknown_s,
                "precedence": "later JUnit overrides earlier for the same identity",
                "excludes": "collection, imports, provenance, examples, contention",
                "speedup": None},
            "shard_timeout_s": shard_timeout,
            "collection_timeout_s": deadline(collection_timeout),
            "source_consistency": consistent(repo, report["source"]),
            "verdict": "PASS",
        })
        validate_plan(report)
    except Exception as error:
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"] = now()
        save(artifacts / "plan.json", report)
        print(f"OPENOPT_SHARD_PLAN verdict={report['verdict']} plan={artifacts / 'plan.json'} "
              f"sha256={digest(artifacts / 'plan.json')}", flush=True)
    return artifacts


def accept_shard(plan, item, artifacts):
    process = read(artifacts / "process.json")
    require(process["returncode"] == 0 and process["timeout"] is False,
            "shard_process_not_pass")
    require(process["command"] == shard_command(Path(plan["repo"]), item, artifacts),
            "shard_process_command_mismatch")
    require(process["timeout_s"] == plan["shard_timeout_s"]
            and type(process["pid"]) is int and process["pid"] > 0
            and math.isfinite(process["wall_s"]) and process["wall_s"] > 0,
            "invalid_shard_process_measurement")
    reports = list(artifacts.glob("tests-*/verification.json"))
    require(len(reports) == 1, "missing_or_multiple_verification_reports")
    path = reports[0]
    report = read(path)
    require(report["verdict"] == "PASS", "verification_not_pass")
    require(report["mode"] == "tests" and report["targets"] == item["targets"]
            and report["coverage"] == "explicit_targets", "verification_target_mismatch")
    require(report["source"] == plan["source"], "shard_source_mismatch")
    expected = verifier(Path(plan["repo"])).check_source_consistency(plan["source"], plan["source"])
    require(report["source_consistency"] == expected, "shard_source_consistency_not_pass")
    xml_path = path.parent / "pytest.xml"
    records, counts = parse_junit(xml_path)
    require(counts == report["tests"], "verification_junit_count_mismatch")
    require(not counts["errors"] and not counts["failures"], "shard_junit_not_pass")
    mapping = {junit_key(node): node for node in plan["collection"]["nodeids"]}
    require(all(record["key"] in mapping for record in records), "unexpected_junit_identity")
    actual = [mapping[record["key"]] for record in records]
    require(Counter(actual) == Counter(item["nodeids"]), "shard_junit_nodeid_mismatch")
    require(report["cuda_available"] is False, "cpu_regression_observed_cuda")
    return {
        "id": item["id"], "verdict": "PASS", "nodeids": actual, "tests": counts,
        "artifact_dir": str(artifacts), "verification": str(path),
        "verification_sha256": digest(path), "junit": str(xml_path),
        "junit_sha256": digest(xml_path), "process_sha256": digest(artifacts / "process.json"),
        "predicted_test_s": item["predicted_test_s"],
        "actual_test_s": sum(record["time_s"] for record in records),
        "actual_worker_wall_s": process["wall_s"], "source": report["source"],
    }


def shard_command(repo, item, artifacts):
    command = [sys.executable, "-B", str(repo / "scripts/verify.py"),
               "tests", "--artifact-dir", str(artifacts)]
    for target in item["targets"]:
        command.extend(["--target", target])
    return command


def launch_shard(repo, plan, item, artifacts):
    consistent(repo, plan["source"])
    command = shard_command(repo, item, artifacts)
    command_logged(command, repo, artifacts, plan["shard_timeout_s"], cpu_env(repo, artifacts))
    consistent(repo, plan["source"])
    return accept_shard(plan, item, artifacts)


def recollect_plan(repo, plan, directory):
    version = plan["schema_version"]
    nodes = collect(repo, directory, plan["collection_timeout_s"],
                    FIXTURE_CONTRACT if version == 2 else "file",
                    plan["grouping"]["selected_files"] if version == 2 else ())
    require(nodes == plan["collection"]["nodeids"], "collection_drift")
    if version == 2:
        fresh = read(directory / "collection.json")
        require(fresh.get("fixture_audit") == plan["grouping"]["audit"],
                "fixture_audit_drift")


def combine(plan, results):
    require(len(results) == len(plan["shards"]), "partial_shard_results")
    require(sorted(item["id"] for item in results) == list(range(len(plan["shards"]))),
            "duplicate_or_missing_shard_results")
    nodes = [node for item in results for node in item["nodeids"]]
    require(Counter(nodes) == Counter(plan["collection"]["nodeids"]), "full_collection_mismatch")
    require(len(nodes) == len(set(nodes)), "duplicate_executed_nodeid")
    counts = {key: sum(item["tests"][key] for item in results)
              for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] == len(nodes) and not counts["failures"] and not counts["errors"],
            "combined_junit_not_pass")
    return counts


def execute(repo, plan_path, plan_pin, artifact_root, workers=3):
    capacity(workers, "workers")
    artifacts = new_artifacts(artifact_root, repo, "run-")
    started = time.monotonic()
    summary = {
        "schema_version": 1, "mode": "tests", "coverage": "tests", "targets": ["tests"],
        "qualification": "full CPU correctness only; not CUDA, commercial or performance qualification",
        "started_at": now(), "verdict": "RUNNING", "workers": workers,
        "plan": str(plan_path.resolve()), "plan_sha256": plan_pin, "shards": [],
        "failures": [], "not_admitted": [], "speedup": None,
    }
    try:
        plan = read(plan_path, plan_pin)
        validate_plan(plan)
        summary["plan_schema_version"] = plan["schema_version"]
        require(Path(plan["repo"]).resolve() == repo, "plan_repo_mismatch")
        summary["source"] = plan["source"]
        consistent(repo, plan["source"])
        directory = artifacts / "collection"
        directory.mkdir()
        recollect_plan(repo, plan, directory)
        pending = iter(plan["shards"])
        admitted = set()
        stop = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            flying = {}

            def admit():
                item = next(pending, None)
                if item is None:
                    return False
                folder = artifacts / f"shard-{item['id']}"
                folder.mkdir()
                admitted.add(item["id"])
                flying[pool.submit(launch_shard, repo, plan, item, folder)] = item
                return True

            for _ in range(min(workers, len(plan["shards"]))):
                admit()
            while flying:
                finished, _ = wait(flying, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = flying.pop(future)
                    try:
                        summary["shards"].append(future.result())
                    except Exception as error:
                        stop = True
                        summary["failures"].append({
                            "id": item["id"], "error": f"{type(error).__name__}: {error}",
                            "artifact_dir": str(artifacts / f"shard-{item['id']}"),
                        })
                # Inspect all completions before admitting any queued work.
                if not stop:
                    for _ in finished:
                        if not admit():
                            break
        summary["not_admitted"] = [item["id"] for item in plan["shards"]
                                   if item["id"] not in admitted]
        if summary["failures"]:
            raise GateError(f"shard_failure: {summary['failures'][0]}")
        summary["tests"] = combine(plan, summary["shards"])
        summary["source_consistency"] = consistent(repo, plan["source"])
        summary["verdict"] = "PASS"
        combined = ET.Element("testsuites", name="openopt full CPU sharded tests")
        for item in sorted(summary["shards"], key=lambda value: value["id"]):
            root = ET.parse(item["junit"]).getroot()
            for suite in ([root] if root.tag == "testsuite" else root.findall("testsuite")):
                combined.append(suite)
        ET.ElementTree(combined).write(artifacts / "pytest.xml", encoding="utf-8",
                                      xml_declaration=True)
        summary["junit_sha256"] = digest(artifacts / "pytest.xml")
    except Exception as error:
        summary.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(finished_at=now(), actual_total_wall_s=time.monotonic() - started)
        save(artifacts / "verification.json", summary)
        print(f"OPENOPT_SHARD_RUN verdict={summary['verdict']} "
              f"report={artifacts / 'verification.json'} "
              f"sha256={digest(artifacts / 'verification.json')}", flush=True)
    return artifacts


def check(repo, plan_path, plan_pin, report_path, report_pin, artifact_root):
    artifacts = new_artifacts(artifact_root, repo, "check-")
    audit = {"schema_version": 1, "started_at": now(), "verdict": "RUNNING"}
    try:
        plan, report = read(plan_path, plan_pin), read(report_path, report_pin)
        validate_plan(plan)
        require(Path(plan["repo"]).resolve() == repo, "plan_repo_mismatch")
        require(report["verdict"] == "PASS" and report["plan_sha256"] == plan_pin,
                "incomplete_or_unbound_run")
        require(Path(report["plan"]).resolve() == plan_path.resolve(), "report_plan_mismatch")
        require(report["coverage"] == "tests" and report["source"] == plan["source"],
                "report_source_or_coverage_mismatch")
        require(not report["failures"] and not report["not_admitted"], "hidden_partial_failure")
        require(len(report["shards"]) == len(plan["shards"]), "partial_shard_results")
        consistent(repo, plan["source"])
        require(read(plan["collection"]["path"], plan["collection"]["sha256"])["nodeids"]
                == plan["collection"]["nodeids"], "plan_collection_sidecar_mismatch")
        fresh = artifacts / "collection"
        fresh.mkdir()
        recollect_plan(repo, plan, fresh)
        replayed = []
        for item in report["shards"]:
            folder = report_path.parent / f"shard-{item['id']}"
            require(Path(item["artifact_dir"]).resolve() == folder.resolve(),
                    "shard_artifact_location_mismatch")
            require(digest(folder / "process.json") == item["process_sha256"],
                    "process_pin_mismatch")
            require(digest(item["verification"]) == item["verification_sha256"]
                    and digest(item["junit"]) == item["junit_sha256"], "shard_pin_mismatch")
            actual = accept_shard(plan, plan["shards"][item["id"]], folder)
            require(actual == item, "shard_replay_mismatch")
            replayed.append(actual)
        counts = combine(plan, replayed)
        require(counts == report["tests"], "combined_report_count_mismatch")
        require(digest(report_path.parent / "pytest.xml") == report["junit_sha256"],
                "combined_junit_pin_mismatch")
        records, xml_counts = parse_junit(report_path.parent / "pytest.xml")
        require(xml_counts == counts and Counter(record["key"] for record in records)
                == Counter(junit_key(node) for node in plan["collection"]["nodeids"]),
                "combined_junit_membership_mismatch")
        audit.update(tests=counts, coverage="tests", source=plan["source"],
                     source_consistency=consistent(repo, plan["source"]), verdict="PASS")
        require(report["source_consistency"] == audit["source_consistency"],
                "report_source_consistency_mismatch")
    except Exception as error:
        audit.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        audit["finished_at"] = now()
        save(artifacts / "acceptance.json", audit)
        print(f"OPENOPT_SHARD_CHECK verdict={audit['verdict']} "
              f"report={artifacts / 'acceptance.json'}", flush=True)
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "check"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", "--log-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--history-junit", type=Path, action="append", default=[])
    parser.add_argument("--shards", type=int, default=3,
                        help="CPU partitions (1-32); independent of GPU limits")
    parser.add_argument("--workers", type=int, default=3,
                        help="simultaneous CPU workers (1-32); may be fewer than shards")
    parser.add_argument("--unknown-test-seconds", type=float, default=1.0)
    parser.add_argument("--shard-timeout", type=float, default=300)
    parser.add_argument("--collection-timeout", type=float, default=60)
    parser.add_argument("--fixture-aware", action="store_true",
                        help="opt in to v2 audited fixture components; unknown scope fails closed")
    parser.add_argument("--fixture-file", action="append", default=[],
                        help="explicit collected file to audit/split; others remain whole files")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-sha256")
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    try:
        if args.action == "plan":
            plan_run(repo, args.artifact_dir, args.history_junit, args.shards,
                     args.unknown_test_seconds, args.shard_timeout, args.collection_timeout,
                     args.fixture_aware, args.fixture_file)
        else:
            require(not args.fixture_aware and not args.fixture_file,
                    "fixture_opt_in_is_plan_only_use_pinned_contract")
            require(args.plan is not None and args.plan_sha256 is not None,
                    "caller_pinned_plan_required")
            if args.action == "run":
                execute(repo, args.plan, args.plan_sha256, args.artifact_dir, args.workers)
            else:
                require(args.report is not None and args.report_sha256 is not None,
                        "caller_pinned_report_required")
                check(repo, args.plan, args.plan_sha256, args.report, args.report_sha256,
                      args.artifact_dir)
    except Exception as error:
        print(f"OPENOPT_SHARDED error={type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

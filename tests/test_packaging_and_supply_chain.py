from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import reflex
import reflex.cli
import system1
import system1.cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRS = (REPO_ROOT / "src" / "system1", REPO_ROOT / "src" / "reflex")


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for src_dir in SRC_DIRS:
        files.extend(sorted(src_dir.rglob("*.py")))
    return files


def _call_qualname(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        cur = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def test_ast_supply_chain_no_unsafe_deserialization_or_trust_remote_code() -> None:
    forbidden_modules = {
        "pickle",
        "_pickle",
        "marshal",
        "shelve",
        "dill",
        "cloudpickle",
    }
    forbidden_calls = {
        "pickle.load",
        "pickle.loads",
        "marshal.load",
        "marshal.loads",
        "shelve.open",
        "eval",
        "exec",
    }
    source_files = _iter_source_files()
    assert len(source_files) > 0

    for py_file in source_files:
        rel = py_file.relative_to(REPO_ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in forbidden_modules, f"Forbidden import {alias.name!r} in {rel}:{node.lineno}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    assert root_mod not in forbidden_modules, f"Forbidden import from {node.module!r} in {rel}:{node.lineno}"
            if not isinstance(node, ast.Call):
                continue
            qname = _call_qualname(node)
            assert qname not in forbidden_calls, f"Forbidden call {qname!r} in {rel}:{node.lineno}"

            for kw in node.keywords:
                if kw.arg == "trust_remote_code":
                    is_true = isinstance(kw.value, ast.Constant) and kw.value.value is True
                    assert not is_true, f"trust_remote_code=True is forbidden in {rel}:{node.lineno}"

            if qname in {"np.load", "numpy.load"}:
                allow_pickle_kws = [kw for kw in node.keywords if kw.arg == "allow_pickle"]
                assert len(allow_pickle_kws) == 1, f"np.load without allow_pickle=False in {rel}:{node.lineno}"
                kw_val = allow_pickle_kws[0].value
                assert (
                    isinstance(kw_val, ast.Constant) and kw_val.value is False
                ), f"np.load must pass literal allow_pickle=False in {rel}:{node.lineno}"


def test_pyproject_metadata_and_entrypoints_contract() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    project = data["project"]
    assert project["name"] == "system1"
    assert project["requires-python"] == ">=3.11"

    scripts = project["scripts"]
    assert scripts["system1"] == "system1.cli:main"
    assert scripts["reflex"] == "reflex.cli:main"

    packages_find = data["tool"]["setuptools"]["packages"]["find"]
    assert packages_find["where"] == ["src"]
    assert "system1*" in packages_find["include"]
    assert "reflex*" in packages_find["include"]

    deps_joined = " ".join(project["dependencies"]).lower()
    for required_dep in ("numpy", "cryptography"):
        assert required_dep in deps_joined

    optional_deps = project["optional-dependencies"]
    assert "gameboy" not in optional_deps
    for expected_extra in ("teaching", "neural", "dev", "grpc"):
        assert expected_extra in optional_deps



def test_cli_subcommand_inventory_and_reflex_parity() -> None:
    expected_subcommands = {
        "decide",
        "bench",
        "calibrate",
        "eval",
        "verify-receipt",
        "compile",
        "teach",
        "serve",
    }

    for parser in (system1.cli.build_parser(), reflex.cli.build_parser()):
        subparsers_actions = [
            action
            for action in parser._actions
            if getattr(action, "choices", None) and isinstance(action.choices, dict)
        ]
        assert len(subparsers_actions) == 1
        assert set(subparsers_actions[0].choices.keys()) == expected_subcommands

    for sym in (
        "evaluate",
        "EvaluationReport",
        "FieldEvaluationMetrics",
        "RecalibrationMetrics",
        "RecalibrationReport",
    ):
        assert getattr(system1, sym) is getattr(reflex, sym)

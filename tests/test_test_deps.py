"""1.2.1 R7: hypothesis is a required test dependency. No test file may import it behind try/except ImportError (or
importorskip), so property tests can never disappear silently; without hypothesis, collection fails loudly."""

import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")


def _imports_hypothesis(nodes):
    for n in nodes:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Import) and any(a.name.split(".")[0] == "hypothesis" for a in sub.names):
                return True
            if isinstance(sub, ast.ImportFrom) and (sub.module or "").split(".")[0] == "hypothesis":
                return True
    return False


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _guarded_imports(path):
    tree = ast.parse(_read(path), path)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Try) and _imports_hypothesis(n.body):
            out.append(f"{os.path.basename(path)}:{n.lineno}: hypothesis imported inside try/except")
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "importorskip":
            if n.args and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith("hypothesis"):
                out.append(f"{os.path.basename(path)}:{n.lineno}: pytest.importorskip('hypothesis')")
    return out


def test_hypothesis_is_never_optional_in_tests():
    files = sorted(os.path.join(TESTS, f) for f in os.listdir(TESTS) if f.endswith(".py"))
    users = [f for f in files if "hypothesis" in _read(f)]
    assert len(users) >= 5  # the property-test files are really scanned
    bad = [b for f in users for b in _guarded_imports(f)]
    assert bad == []


def test_hypothesis_is_in_the_test_extra():
    text = _read(os.path.join(ROOT, "pyproject.toml"))
    test_extra = re.search(r"^test = \[([^\]]*)\]", text, re.M)
    assert test_extra and re.search(r'"hypothesis[>=<~ ]', test_extra.group(1)), "hypothesis missing from [test]"

"""1.2.1 R5: only the version being prepared may be marked unreleased in CHANGELOG.md; every older entry is dated."""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_only_the_current_version_is_unreleased():
    version = re.search(r'^__version__ = "([^"]+)"', _read("src/roman_wheel_node/_version.py"), re.M).group(1)
    heads = re.findall(r"^## (\S+) \(([^)]*)\)", _read("CHANGELOG.md"), re.M)
    assert heads and heads[0][0] == version, heads[:2]  # newest entry first, and it is the package version
    for v, when in heads[1:]:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", when), f"CHANGELOG entry {v} is ({when}), not a release date"

"""`python -m roman_wheel_node`: --help, --version, the default demo, and a non-zero exit on bad arguments."""

import subprocess
import sys

import roman_wheel_node as pkg


def run(*args):
    return subprocess.run([sys.executable, "-m", "roman_wheel_node", *args], capture_output=True, text=True, timeout=60)


def test_version():
    r = run("--version")
    assert r.returncode == 0 and r.stdout.strip() == f"roman-wheel-node {pkg.__version__}"


def test_help():
    r = run("--help")
    assert r.returncode == 0 and "usage: roman-wheel-node" in r.stdout and "demo" in r.stdout


def test_default_demo_prints_the_six_reference_calls():
    for args in ((), ("demo",)):
        r = run(*args)
        assert r.returncode == 0, r.stderr
        lines = r.stdout.splitlines()
        assert len(lines) == 6
        assert lines[2].split()[0] == "0" and "BLOCK" in lines[2]


def test_bad_arguments_exit_non_zero():
    for args in (("--nope",), ("frobnicate",), ("demo", "extra")):
        r = run(*args)
        assert r.returncode == 2 and "usage:" in r.stderr, args

"""Phase 5 GPU check, run by phase5/setup_gpu.ps1 after the hash-locked torch install.

Fails loudly with a non-zero exit code on anything short of a working cu132 build on an
sm_120-capable GPU. The matmul self-test uses a fixed deterministic tensor and compares the
GPU result with the same product computed on the CPU, so it is reproducible run to run.

Exit codes: 0 ok, 2 torch missing, 3 wrong torch build, 4 CUDA not available,
5 no kernels for this GPU, 6 GPU matmul differs from CPU.
"""
import sys

EXPECTED_TORCH = "2.14.1+cu132"
EXPECTED_CUDA = "13.2"
N = 64


def fail(code, msg):
    print("FAIL: " + msg, file=sys.stderr)
    sys.exit(code)


def fixed_matrix(torch):
    # Small exact values (k/4 with k in -3..3), so every product and sum is exact in float32
    # and the GPU result must match the CPU result bit for bit.
    return ((torch.arange(N * N, dtype=torch.int64) % 7) - 3).to(torch.float32).reshape(N, N) / 4


def main():
    try:
        import torch
    except ImportError as e:
        fail(2, f"torch is not importable: {e}")
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_CUDA:
        fail(3, f"expected torch {EXPECTED_TORCH} (CUDA {EXPECTED_CUDA}), got {torch.__version__} "
             f"(CUDA {torch.version.cuda}); rerun phase5\\setup_gpu.ps1")
    if not torch.cuda.is_available():
        fail(4, "CUDA not available: check the NVIDIA driver (nvidia-smi must report CUDA 13.2 or newer)")
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    print("gpu", torch.cuda.get_device_name(0), "capability", (major, minor), "arch", arch)
    if arch not in arch_list:
        fail(5, f"this torch build has no kernels for {arch}: {arch_list}")
    x = fixed_matrix(torch)
    want = x @ x
    got = (x.to("cuda") @ x.to("cuda")).cpu()
    if not torch.equal(got, want):
        fail(6, f"GPU matmul differs from CPU: max abs diff {float((got - want).abs().max()):g}")
    print(f"gpu matmul ok (matches CPU exactly, checksum {float(want.sum()):g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

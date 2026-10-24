# Phase 5 GPU setup for RTX 50-series (Blackwell, sm_120) on Windows.
# Installs the official prebuilt PyTorch built against CUDA 13.2. The CUDA runtime is
# bundled inside the wheel, so no separate CUDA toolkit install is needed. Any NVIDIA
# driver that reports CUDA 13.2 or newer in nvidia-smi (e.g. 13.5) runs it.
# Usage (venv active, from the repo root):  .\phase5\setup_gpu.ps1
$ErrorActionPreference = "Stop"
$TorchVersion = "2.14.1"
$Index = "https://download.pytorch.org/whl/cu132"

python -m pip uninstall -y torch torchvision torchaudio
python -m pip install "torch==$TorchVersion" --index-url $Index
python -m pip install -e ".[test,local-nli2]"

python -c @"
import torch, sys
print('torch', torch.__version__, 'cuda', torch.version.cuda)
if not torch.cuda.is_available(): sys.exit('CUDA not available: check the NVIDIA driver (nvidia-smi)')
cap = torch.cuda.get_device_capability(0)
print('gpu', torch.cuda.get_device_name(0), 'capability', cap)
if 'sm_%d%d' % cap not in torch.cuda.get_arch_list(): sys.exit('this torch build has no kernels for your GPU: %s' % torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device='cuda'); print('gpu matmul ok', float((x @ x).sum()) == float((x @ x).sum()))
"@
Write-Host "GPU ready. Next: `$env:PYTHONPATH = 'src'; python phase5\train_pair_checker.py --out phase5\models\pairs-v1"

import subprocess
import sys
import psutil

print("=== CHECKING NVIDIA GPU & DRIVER ===")
try:
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if res.returncode == 0:
        print("nvidia-smi Output:\n" + res.stdout)
    else:
        print(f"nvidia-smi failed (code {res.returncode}):\n{res.stderr}")
except Exception as e:
    print(f"Error running nvidia-smi: {e}")

print("\n=== CHECKING PYTORCH & CUDA ===")
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_avail}")
    if cuda_avail:
        print(f"Device Count:    {torch.cuda.device_count()}")
        print(f"Device 0 Name:   {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version:    {torch.version.cuda}")
        print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA is not currently reporting available to PyTorch.")
except Exception as e:
    print(f"Error importing torch: {e}")

print("\n=== DISK SPACE CHECK ===")
for part in psutil.disk_partitions():
    try:
        usage = psutil.disk_usage(part.mountpoint)
        total_gb = usage.total / (1024**3)
        free_gb = usage.free / (1024**3)
        used_gb = usage.used / (1024**3)
        pct = usage.percent
        print(f"Drive {part.mountpoint:10s} Total: {total_gb:6.1f} GB | Used: {used_gb:6.1f} GB ({pct:4.1f}%) | Free: {free_gb:6.1f} GB")
    except Exception:
        pass

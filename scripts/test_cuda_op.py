import torch

print("Testing CUDA allocation on RTX 5050 (sm_120)...")
try:
    x = torch.randn(10, 10, device="cuda")
    y = x @ x
    print("Success! CUDA tensor operation completed:\n", y[:2, :2])
except Exception as e:
    print(f"CUDA operation failed on sm_120: {e}")

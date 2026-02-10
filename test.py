import torch

print(torch.cuda.is_bf16_supported(including_emulation=False))
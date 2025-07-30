from rust_packer import packb
from rust_packer import unpackb
import torch


def run_test():
    original_tensor = torch.zeros(10, 5, dtype=torch.float32)
    packed_with_rust = packb(original_tensor)
    _ = unpackb(packed_with_rust)

if __name__ == "__main__":
    # NOTE: You'll need to save your original Python code to a file
    # named `your_original_packer.py` for this to run.
    run_test()
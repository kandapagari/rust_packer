# test.py

from rust_packer import packb as rust_packb
from rust_packer import unpackb as rust_unpackb
from tensor_pack import packb as python_packb
from tensor_pack import unpackb as python_unpackb
import torch


def run_test():
    print("🚀 Starting Rust vs. Python Packer Test...")

    # 1. Create a sample tensor
    # Try different dtypes like torch.float32, torch.int64, or even torch.bfloat16
    original_tensor = torch.randn(10, 5, dtype=torch.float32)
    print(f"\nOriginal Tensor (dtype: {original_tensor.dtype}):\n{original_tensor.shape}")

    # 2. Pack with both implementations
    print("\n--- Packing ---")
    packed_with_python = python_packb(original_tensor)
    packed_with_rust = rust_packb(original_tensor)

    print(f"Python packed size: {len(packed_with_python)} bytes")
    print(f"Rust packed size:   {len(packed_with_rust)} bytes")

    # 3. Unpack with both implementations
    print("\n--- Unpacking ---")
    unpacked_from_python_bytes = python_unpackb(packed_with_python)
    unpacked_from_rust_bytes = rust_unpackb(packed_with_rust)

    # 4. Verify the results
    assert torch.equal(original_tensor, unpacked_from_python_bytes)
    assert torch.equal(original_tensor, unpacked_from_rust_bytes)
    print("✅ Unpacked tensors are identical to the original.")

    print("\n🎉 Test finished successfully!")


if __name__ == "__main__":
    # NOTE: You'll need to save your original Python code to a file
    # named `your_original_packer.py` for this to run.
    run_test()

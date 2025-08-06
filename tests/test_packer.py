# tests/test_packer.py

import pytest
import numpy as np
import torch
from rust_packer import packb, unpackb # type: ignore


# Helper function to compare tensors/arrays
def assert_equal(a, b):
    """Asserts that two tensors or numpy arrays are equal."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        # For GPU tensors, move to CPU before comparing values
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        assert np.array_equal(a, b)
    else:
        pytest.fail(f"Type mismatch: {type(a)} vs {type(b)}")


# --- Test Cases for NumPy Arrays ---


def test_numpy_roundtrip():
    """
    Tests packing and unpacking a simple NumPy ndarray.
    """
    # 1. Create the original numpy array
    original_array = np.arange(12, dtype=np.float32).reshape(3, 4)

    # 2. Pack the array using the Rust function
    packed_bytes = packb(original_array)
    assert isinstance(packed_bytes, bytes)

    # 3. Unpack the bytes
    unpacked_array = unpackb(packed_bytes)

    # 4. Verify the result is identical to the original
    assert isinstance(unpacked_array, np.ndarray)
    assert_equal(original_array, unpacked_array)


# --- Test Cases for PyTorch Tensors ---


def test_torch_cpu_roundtrip():
    """
    Tests packing and unpacking a simple PyTorch Tensor on the CPU.
    """
    original_tensor = torch.arange(12, dtype=torch.float64).reshape(3, 4)

    packed_bytes = packb(original_tensor)
    unpacked_tensor = unpackb(packed_bytes)

    assert isinstance(unpacked_tensor, torch.Tensor)
    assert_equal(original_tensor, unpacked_tensor)


def test_torch_bfloat16_roundtrip():
    """
    Tests the special handling for bfloat16 tensors.
    """
    original_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)

    packed_bytes = packb(original_tensor)
    unpacked_tensor = unpackb(packed_bytes)

    assert isinstance(unpacked_tensor, torch.Tensor)
    # The main check: did it come back as bfloat16?
    assert unpacked_tensor.dtype == torch.bfloat16
    assert_equal(original_tensor, unpacked_tensor)


def test_torch_non_contiguous_roundtrip():
    """
    Tests that non-contiguous tensors are correctly handled.
    Your Rust code explicitly makes tensors contiguous before serialization.
    """
    # A transposed tensor is a common way to get a non-contiguous view
    contiguous_tensor = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    non_contiguous_tensor = contiguous_tensor.T

    assert not non_contiguous_tensor.is_contiguous()  # Sanity check

    packed_bytes = packb(non_contiguous_tensor)
    unpacked_tensor = unpackb(packed_bytes)

    # The unpacked tensor should be equal to the non-contiguous one in values,
    # but it will be contiguous itself.
    assert unpacked_tensor.is_contiguous()
    assert_equal(non_contiguous_tensor, unpacked_tensor)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PyTorch CUDA is not available"
)
def test_torch_gpu_roundtrip():
    """
    Tests packing a GPU tensor. The result should be a CPU tensor.
    This test is skipped if a CUDA-enabled GPU is not found.
    """
    original_tensor_gpu = torch.randn(2, 2, device="cuda")

    packed_bytes = packb(original_tensor_gpu)
    unpacked_tensor = unpackb(packed_bytes)

    # Your code moves tensors to the CPU before packing, so the unpacked
    # tensor should be on the CPU but have the same data.
    assert isinstance(unpacked_tensor, torch.Tensor)
    assert unpacked_tensor.device.type == "cpu"
    assert_equal(original_tensor_gpu, unpacked_tensor)


# --- Test Cases for Error Handling ---


def test_pack_invalid_type():
    """
    Tests that packb raises a TypeError for unsupported Python types.
    """
    not_a_tensor = [1, 2, 3]  # A simple list
    with pytest.raises(TypeError, match="Expected a Tensor or ndarray, got list"):
        packb(not_a_tensor)


def test_unpack_invalid_bytes():
    """
    Tests that unpackb raises a ValueError for malformed byte input.
    """
    invalid_bytes = b"this is not a valid msgpack bytestring"
    with pytest.raises(ValueError, match="Deserialization error"):
        unpackb(invalid_bytes)

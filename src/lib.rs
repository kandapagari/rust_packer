// src/lib.rs

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{PyAny, PyResult};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SerializableTensor {
    #[serde(rename = "__torch_tensor__")]
    is_torch: bool,
    data: Vec<u8>,
    dtype: String,
    shape: Vec<usize>,
    original_dtype: String,
}

#[pyfunction]
fn packb(obj: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let py = obj.py();

    let binding = obj.get_type();
    let type_name = binding.name()?;
    
    if type_name != "Tensor" {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a Tensor, got {}",
            type_name
        )));
    }
    let device_type: String = obj.getattr("device")?.getattr("type")?.extract()?;

    let obj_cpu = if device_type != "cpu" {
        obj.call_method0("cpu")?
    } else {
        obj.to_owned()
    };

    let is_contiguous: bool = obj_cpu.call_method0("is_contiguous")?.extract()?;
    let obj_contiguous = if !is_contiguous {
        obj_cpu.call_method0("contiguous")?
    } else {
        obj_cpu
    };

    let original_dtype = obj_contiguous.getattr("dtype")?.str()?.to_str()?.replace("torch.", "");
    let obj_final = if original_dtype == "bfloat16" {
        let torch = PyModule::import(py, "torch")?;
        let float32_dtype = torch.getattr("float32")?;
        obj_contiguous.call_method1("to", (float32_dtype,))?
    } else {
        obj_contiguous
    };

    let numpy_array = obj_final.call_method0("numpy")?;

    let data_bytes: Bound<'_, PyBytes> = numpy_array.call_method0("tobytes")?.extract()?;

    let shape: Vec<usize> = numpy_array.getattr("shape")?.extract()?;
    let dtype_str: String = numpy_array.getattr("dtype")?.getattr("str")?.extract()?;

    let serializable = SerializableTensor {
        is_torch: true,
        data: data_bytes.as_bytes().to_vec(),
        dtype: dtype_str,
        shape,
        original_dtype,
    };

    let packed = rmp_serde::to_vec_named(&serializable).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e))
    })?;

    Ok(PyBytes::new(py, &packed).into())
}

#[pyfunction]
fn unpackb(bytes: &Bound<'_, PyBytes>) -> PyResult<Py<PyAny>> {
    let py = bytes.py();
    let data: &[u8] = bytes.as_bytes();

    let unpacked: SerializableTensor = rmp_serde::from_slice(data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Deserialization error: {}", e))
    })?;

    let np = PyModule::import(py, "numpy")?;
    let torch = PyModule::import(py, "torch")?;

    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", unpacked.dtype)?;

    let py_array = np
        .call_method(
            "frombuffer",
            (PyBytes::new(py, &unpacked.data),),
            Some(&kwargs),
        )?
        .call_method("reshape", (unpacked.shape,), None)?
        .call_method0("copy")?;

    let mut tensor = torch.call_method1("as_tensor", (py_array,))?;

    if unpacked.original_dtype == "bfloat16" {
        let bfloat16_dtype = torch.getattr("bfloat16")?;
        tensor = tensor.call_method1("to", (bfloat16_dtype,))?;
    }

    Ok(tensor.into())
}

#[pymodule]
fn rust_packer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(packb, m)?)?;
    m.add_function(wrap_pyfunction!(unpackb, m)?)?;
    Ok(())
}

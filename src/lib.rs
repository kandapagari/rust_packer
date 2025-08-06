// src/lib.rs

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{PyAny, PyResult};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SerializableTensor {
    #[serde(rename = "__torch_tensor__")]
    pub is_torch: bool,
    pub data: Vec<u8>,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub original_dtype: String,
}

#[pyfunction]
pub fn packb(obj: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let py: Python<'_> = obj.py();

    let binding: Bound<'_, pyo3::types::PyType> = obj.get_type();
    let type_name: Bound<'_, pyo3::types::PyString> = binding.name()?;

    if type_name == "Tensor" {
        let device_type: String = obj.getattr("device")?.getattr("type")?.extract()?;
        let obj_cpu: Bound<'_, PyAny> = if device_type != "cpu" {
            obj.call_method0("cpu")?
        } else {
            obj.to_owned()
        };
        let is_contiguous: bool = obj_cpu.call_method0("is_contiguous")?.extract()?;
        let obj_contiguous: Bound<'_, PyAny> = if !is_contiguous {
            obj_cpu.call_method0("contiguous")?
        } else {
            obj_cpu
        };
        let original_dtype: String = obj_contiguous
            .getattr("dtype")?
            .str()?
            .to_str()?
            .replace("torch.", "");
        let obj_final: Bound<'_, PyAny> = if original_dtype == "bfloat16" {
            let torch: Bound<'_, PyModule> = PyModule::import(py, "torch")?;
            let float32_dtype: Bound<'_, PyAny> = torch.getattr("float32")?;
            obj_contiguous.call_method1("to", (float32_dtype,))?
        } else {
            obj_contiguous
        };
        let numpy_array: Bound<'_, PyAny> = obj_final.call_method0("numpy")?;
        let data_bytes: Bound<'_, PyBytes> = numpy_array.call_method0("tobytes")?.extract()?;
        let shape: Vec<usize> = numpy_array.getattr("shape")?.extract()?;
        let dtype_str: String = numpy_array.getattr("dtype")?.getattr("str")?.extract()?;
        let serializable: SerializableTensor = SerializableTensor {
            is_torch: true,
            data: data_bytes.as_bytes().to_vec(),
            dtype: dtype_str,
            shape,
            original_dtype,
        };
        let packed: Vec<u8> = rmp_serde::to_vec_named(&serializable).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e))
        })?;
        Ok(PyBytes::new(py, &packed).into())
    } else if type_name == "ndarray" {
        let data_bytes: Bound<'_, PyBytes> = obj.call_method0("tobytes")?.extract()?;
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
        let dtype_str: String = obj.getattr("dtype")?.getattr("str")?.extract()?;
        let serializable: SerializableTensor = SerializableTensor {
            is_torch: false,
            data: data_bytes.as_bytes().to_vec(),
            dtype: dtype_str.clone(),
            shape,
            original_dtype: dtype_str,
        };
        let packed: Vec<u8> = rmp_serde::to_vec_named(&serializable).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e))
        })?;
        Ok(PyBytes::new(py, &packed).into())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a Tensor or ndarray, got {}",
            type_name
        )))
    }
}

#[pyfunction]
pub fn unpackb(bytes: &Bound<'_, PyBytes>) -> PyResult<Py<PyAny>> {
    let py: Python<'_> = bytes.py();
    let data: &[u8] = bytes.as_bytes();
    let unpacked: SerializableTensor = rmp_serde::from_slice(data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Deserialization error: {}", e))
    })?;
    let np: Bound<'_, PyModule> = PyModule::import(py, "numpy")?;
    if unpacked.is_torch {
        let torch: Bound<'_, PyModule> = PyModule::import(py, "torch")?;
        let kwargs: Bound<'_, pyo3::types::PyDict> = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", unpacked.dtype.clone())?;
        let py_array: Bound<'_, PyAny> = np
            .call_method(
                "frombuffer",
                (PyBytes::new(py, &unpacked.data),),
                Some(&kwargs),
            )?
            .call_method("reshape", (unpacked.shape.clone(),), None)?
            .call_method0("copy")?;
        let mut tensor: Bound<'_, PyAny> = torch.call_method1("as_tensor", (py_array,))?;
        if unpacked.original_dtype == "bfloat16" {
            let bfloat16_dtype: Bound<'_, PyAny> = torch.getattr("bfloat16")?;
            tensor = tensor.call_method1("to", (bfloat16_dtype,))?;
        }
        Ok(tensor.into())
    } else {
        let kwargs: Bound<'_, pyo3::types::PyDict> = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", unpacked.dtype.clone())?;
        let py_array: Bound<'_, PyAny> = np
            .call_method(
                "frombuffer",
                (PyBytes::new(py, &unpacked.data),),
                Some(&kwargs),
            )?
            .call_method("reshape", (unpacked.shape.clone(),), None)?
            .call_method0("copy")?;
        Ok(py_array.into())
    }
}

#[pymodule]
fn rust_packer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(packb, m)?)?;
    m.add_function(wrap_pyfunction!(unpackb, m)?)?;
    Ok(())
}

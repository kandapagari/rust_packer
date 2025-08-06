use rust_packer::SerializableTensor;

#[test]
fn test_serializable_tensor_serde() {
    let tensor = SerializableTensor {
        is_torch: true,
        data: vec![1, 2, 3, 4],
        dtype: "float32".to_string(),
        shape: vec![2, 2],
        original_dtype: "float32".to_string(),
    };

    let serialized: Vec<u8> = rmp_serde::to_vec_named(&tensor).unwrap();
    let deserialized: SerializableTensor = rmp_serde::from_slice(&serialized).unwrap();
    assert_eq!(tensor.is_torch, deserialized.is_torch);
    assert_eq!(tensor.data, deserialized.data);
    assert_eq!(tensor.dtype, deserialized.dtype);
    assert_eq!(tensor.shape, deserialized.shape);
    assert_eq!(tensor.original_dtype, deserialized.original_dtype);
}
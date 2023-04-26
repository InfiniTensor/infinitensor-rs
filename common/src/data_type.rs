#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DataType {
    UNDEFINED,
    FLOAT,
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    INT64,
    STRING,
    BOOL,
    FLOAT16,
    DOUBLE,
    UINT32,
    UINT64,
    // COMPLEX64,
    // COMPLEX128,
    // BFLOAT16,
}

impl DataType {
    pub const fn size(&self) -> usize {
        use core::mem::size_of;
        match self {
            DataType::UNDEFINED => unreachable!(),
            DataType::FLOAT => size_of::<f32>(),
            DataType::UINT8 => size_of::<u8>(),
            DataType::INT8 => size_of::<i8>(),
            DataType::UINT16 => size_of::<u16>(),
            DataType::INT16 => size_of::<i16>(),
            DataType::INT32 => size_of::<i32>(),
            DataType::INT64 => size_of::<i64>(),
            DataType::STRING => unreachable!(),
            DataType::BOOL => size_of::<bool>(),
            DataType::FLOAT16 => 2,
            DataType::DOUBLE => size_of::<f64>(),
            DataType::UINT32 => size_of::<u32>(),
            DataType::UINT64 => size_of::<u64>(),
        }
    }
}

pub trait AsDataType {
    fn as_data_type() -> DataType;
}

impl AsDataType for f32 {
    fn as_data_type() -> DataType {
        DataType::FLOAT
    }
}

impl AsDataType for u8 {
    fn as_data_type() -> DataType {
        DataType::UINT8
    }
}

impl AsDataType for i8 {
    fn as_data_type() -> DataType {
        DataType::INT8
    }
}

impl AsDataType for u16 {
    fn as_data_type() -> DataType {
        DataType::UINT16
    }
}

impl AsDataType for i16 {
    fn as_data_type() -> DataType {
        DataType::INT16
    }
}

impl AsDataType for i32 {
    fn as_data_type() -> DataType {
        DataType::INT32
    }
}

impl AsDataType for i64 {
    fn as_data_type() -> DataType {
        DataType::INT64
    }
}

impl AsDataType for bool {
    fn as_data_type() -> DataType {
        DataType::BOOL
    }
}

impl AsDataType for f64 {
    fn as_data_type() -> DataType {
        DataType::DOUBLE
    }
}

impl AsDataType for u32 {
    fn as_data_type() -> DataType {
        DataType::UINT32
    }
}

impl AsDataType for u64 {
    fn as_data_type() -> DataType {
        DataType::UINT64
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
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

    #[inline]
    pub const fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::FLOAT
                | DataType::UINT8
                | DataType::INT8
                | DataType::UINT16
                | DataType::INT16
                | DataType::INT32
                | DataType::INT64
                | DataType::BOOL
                | DataType::FLOAT16
                | DataType::DOUBLE
                | DataType::UINT32
                | DataType::UINT64
        )
    }

    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self, DataType::FLOAT | DataType::FLOAT16 | DataType::DOUBLE)
    }

    #[inline]
    pub const fn is_bool(&self) -> bool {
        matches!(self, DataType::BOOL)
    }

    #[inline]
    pub const fn is_bits(&self) -> bool {
        matches!(
            self,
            DataType::UINT8 | DataType::UINT16 | DataType::UINT32 | DataType::UINT64
        )
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

#[macro_export]
macro_rules! invoke_ty {
    ($data_type:expr, $slice:expr, $func:ident) => {
        match ($data_type) {
            DataType::UNDEFINED => unreachable!(),
            DataType::FLOAT => $func::<f32>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const f32,
                    $slice.len() / core::mem::size_of::<f32>(),
                )
            }),
            DataType::UINT8 => $func::<u8>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const u8,
                    $slice.len() / core::mem::size_of::<u8>(),
                )
            }),
            DataType::INT8 => $func::<i8>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const i8,
                    $slice.len() / core::mem::size_of::<i8>(),
                )
            }),
            DataType::UINT16 => $func::<u16>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const u16,
                    $slice.len() / core::mem::size_of::<u16>(),
                )
            }),
            DataType::INT16 => $func::<i16>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const i16,
                    $slice.len() / core::mem::size_of::<i16>(),
                )
            }),
            DataType::INT32 => $func::<i32>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const i32,
                    $slice.len() / core::mem::size_of::<i32>(),
                )
            }),
            DataType::INT64 => $func::<i64>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const i64,
                    $slice.len() / core::mem::size_of::<i64>(),
                )
            }),
            DataType::STRING => todo!(),
            DataType::BOOL => $func::<bool>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const bool,
                    $slice.len() / core::mem::size_of::<bool>(),
                )
            }),
            DataType::FLOAT16 => todo!(),
            DataType::DOUBLE => $func::<f64>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const f64,
                    $slice.len() / core::mem::size_of::<f64>(),
                )
            }),
            DataType::UINT32 => $func::<u32>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const u32,
                    $slice.len() / core::mem::size_of::<u32>(),
                )
            }),
            DataType::UINT64 => $func::<u64>(unsafe {
                core::slice::from_raw_parts(
                    $slice.as_ptr() as *const u64,
                    $slice.len() / core::mem::size_of::<u64>(),
                )
            }),
        }
    };
}

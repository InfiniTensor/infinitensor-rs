use crate::AsDataType;
use std::alloc::Layout;

pub struct Data {
    cpu: Vec<u8>,
}

impl Data {
    #[inline]
    pub const fn empty() -> Self {
        Self { cpu: Vec::new() }
    }

    pub fn cpu<T: AsDataType>(mut data: Vec<T>) -> Self {
        let len = Layout::array::<T>(data.len()).unwrap();
        let cap = Layout::array::<T>(data.capacity()).unwrap();
        let cpu = unsafe { Vec::from_raw_parts(data.as_mut_ptr() as _, len.size(), cap.size()) };
        core::mem::forget(data);
        Self { cpu }
    }

    #[inline]
    pub fn as_slice<T>(&self) -> &[T] {
        let ptr = self.cpu.as_ptr() as *const T;
        let len = self.cpu.len() / core::mem::size_of::<T>();
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

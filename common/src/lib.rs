#![deny(warnings)]

mod data;
#[macro_use]
mod data_type;

pub use data::Data;
pub use data_type::{AsDataType, DataType};

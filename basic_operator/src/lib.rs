#![deny(warnings)]

mod op_type;

pub mod infer;

pub use op_type::{Binary, Compair, OpType, Reduce, Unary};

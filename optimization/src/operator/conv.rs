use graph::linked::{LinkedTensor, Operator};
use std::sync::Arc;

#[repr(transparent)]
pub struct Conv<'a>(&'a Operator);

impl<'a> Conv<'a> {
    #[inline]
    pub const fn new(op: &'a Operator) -> Self {
        Self(op)
    }

    #[inline]
    pub fn input(&self) -> &Arc<LinkedTensor> {
        self.0.inputs.get(0).unwrap()
    }

    #[inline]
    pub fn kernel(&self) -> &Arc<LinkedTensor> {
        self.0.inputs.get(1).unwrap()
    }

    #[inline]
    pub fn dilations(&self) -> &Arc<LinkedTensor> {
        self.0.inputs.get(2).unwrap()
    }

    #[inline]
    pub fn pads(&self) -> &Arc<LinkedTensor> {
        self.0.inputs.get(3).unwrap()
    }

    #[inline]
    pub fn strides(&self) -> &Arc<LinkedTensor> {
        self.0.inputs.get(4).unwrap()
    }

    #[inline]
    pub fn output(&self) -> &Arc<LinkedTensor> {
        self.0.outputs.get(0).unwrap()
    }
}

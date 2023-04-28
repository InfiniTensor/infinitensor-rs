use crate::Tensor;
use basic_operator::Unary;

pub(super) fn infer(ty: Unary, input: &Tensor) -> Tensor {
    use Unary::*;
    match ty {
        Abs | Erf | Neg => {
            if input.dtype.is_numeric() {
                input.clone_info()
            } else {
                panic!("Unary only support numeric type")
            }
        }
          Acos | Acosh | Asin | Asinh | Atan | Atanh // fmt
        |  Cos |  Cosh |  Sin |  Sinh |  Tan |  Tanh
        | Exp  | Log   | Sqrt
        | Ceil | Floor | Round
        | Relu | Sigmoid => {
            if input.dtype.is_float() {
                input.clone_info()
            } else {
                panic!("Unary only support float type")
            }
        }
        Unary::Not => {
            if input.dtype.is_bool() {
                input.clone_info()
            } else {
                panic!("Unary only support bool type")
            }
        }
    }
}

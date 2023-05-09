mod binary;
mod broadcast;
mod compair;
mod gemm;
mod matmul;
mod pool;
mod reduce;
mod reshape;
mod squeeze;
mod unary;
mod unsqueeze;

use crate::Tensor;
use basic_operator::OpType;

pub(crate) fn infer(op_type: OpType, inputs: &[&Tensor]) -> Vec<Tensor> {
    match op_type {
        OpType::Unary(u) => match inputs {
            &[data] => vec![unary::infer(u, data)],
            _ => panic!("Unary operator must have a single input"),
        },
        OpType::Binary(b) => match inputs {
            &[x, y] => vec![binary::infer(b, x, y)],
            _ => panic!("Binary operator must have two inputs"),
        },
        OpType::Reduce(r) => vec![reduce::infer(r, inputs)],
        OpType::Compair(_) => match inputs {
            &[x, y] => vec![compair::infer(x, y)],
            _ => panic!("Compair operator must have two inputs"),
        },
        OpType::Pool(_) => match inputs {
            &[data, kernel, dilations, pads, strides] => {
                vec![pool::infer(data, kernel, dilations, pads, strides)]
            }
            _ => panic!("Pool operator must have 5 inputs"),
        },
        OpType::ArgMax => todo!(),
        OpType::BatchNormalization => todo!(),
        OpType::Bernoulli => todo!(),
        OpType::BlackmanWindow => todo!(),
        OpType::Cast => match inputs {
            &[data, dtype] => {
                assert!(dtype.shape.is_empty(), "Cast dtype must be empty");
                assert!(dtype.data.is_none(), "Cast dtype must be empty");
                vec![Tensor {
                    shape: data.shape.clone(),
                    dtype: dtype.dtype,
                    data: data.data.clone(),
                }]
            }
            _ => panic!("Cast operator must have two inputs"),
        },
        OpType::CastLike => todo!(),
        OpType::Celu => todo!(),
        OpType::CenterCropPad => todo!(),
        OpType::Clip => todo!(),
        OpType::Col2lm => todo!(),
        OpType::Compress => todo!(),
        OpType::Concat => todo!(),
        OpType::ConcatFromSequence => todo!(),
        OpType::ConstantOfShape => todo!(),
        OpType::Conv => todo!(),
        OpType::ConvInteger => todo!(),
        OpType::ConvTranspose => todo!(),
        OpType::CumSum => todo!(),
        OpType::DFT => todo!(),
        OpType::DeformConv => todo!(),
        OpType::DepthToSpace => todo!(),
        OpType::DequantizeLinear => todo!(),
        OpType::Det => todo!(),
        OpType::Dropout => todo!(),
        OpType::DynamicQuantizeLinear => todo!(),
        OpType::Einsum => todo!(),
        OpType::Elu => todo!(),
        OpType::Expand => todo!(),
        OpType::EyeLike => todo!(),
        OpType::Flatten => todo!(),
        OpType::GRU => todo!(),
        OpType::Gather => todo!(),
        OpType::GatherElements => todo!(),
        OpType::GatherND => todo!(),
        OpType::Gemm => match inputs {
            &[a, b, c, alpha, beta, trans_a, trans_b] => {
                vec![gemm::infer(a, b, c, alpha, beta, trans_a, trans_b)]
            }
            _ => panic!("Gemm operator must have three inputs"),
        },
        OpType::GlobalAveragePool => todo!(),
        OpType::GlobalLpPool => todo!(),
        OpType::GlobalMaxPool => todo!(),
        OpType::GridSample => todo!(),
        OpType::GroupNormalization => todo!(),
        OpType::HammingWindow => todo!(),
        OpType::HannWindow => todo!(),
        OpType::HardSigmoid => todo!(),
        OpType::HardSwish => todo!(),
        OpType::Hardmax => todo!(),
        OpType::Identity => match inputs {
            &[x] => vec![x.clone()],
            _ => panic!("Identity operator must have one input"),
        },
        OpType::If => todo!(),
        OpType::InstanceNormalization => todo!(),
        OpType::IsInf => todo!(),
        OpType::IsNaN => todo!(),
        OpType::LRN => todo!(),
        OpType::LSTM => todo!(),
        OpType::LayerNormalization => todo!(),
        OpType::LeakyRelu => match inputs {
            &[data, alpha] => {
                assert!(alpha.is_scalar());
                assert!(alpha.dtype().is_float());
                assert!(data.dtype().is_float());
                vec![data.clone()]
            }
            _ => panic!("LeakyRelu operator must have two inputs"),
        },
        OpType::LogSoftmax => todo!(),
        OpType::Loop => todo!(),
        OpType::LpNormalization => todo!(),
        OpType::MatMul => match inputs {
            &[a, b] => vec![matmul::infer(a, b)],
            _ => panic!("MatMul operator must have two inputs"),
        },
        OpType::MatMulInteger => todo!(),
        OpType::MeanVarianceNormalization => todo!(),
        OpType::MelWeightMatrix => todo!(),
        OpType::Mish => todo!(),
        OpType::Multinomial => todo!(),
        OpType::NegativeLogLikelihoodLoss => todo!(),
        OpType::NonMaxSuppression => todo!(),
        OpType::NonZero => todo!(),
        OpType::OneHot => todo!(),
        OpType::Optional => todo!(),
        OpType::OptionalGetElement => todo!(),
        OpType::OptionalHasElement => todo!(),
        OpType::PRelu => match inputs {
            &[data, alpha] => {
                assert!(alpha.dtype().is_numeric());
                assert_eq!(alpha.dtype(), data.dtype());
                vec![Tensor {
                    shape: broadcast::unidirection(data.shape(), alpha.shape()).unwrap(),
                    dtype: data.dtype(),
                    data: None,
                }]
            }
            _ => panic!("LeakyRelu operator must have two inputs"),
        },
        OpType::Pad => todo!(),
        OpType::QLinearConv => todo!(),
        OpType::QLinearMatMul => todo!(),
        OpType::QuantizeLinear => todo!(),
        OpType::RNN => todo!(),
        OpType::RandomNormal => todo!(),
        OpType::RandomNormalLike => todo!(),
        OpType::RandomUniform => todo!(),
        OpType::RandomUniformLike => todo!(),
        OpType::Range => todo!(),
        OpType::Reciprocal => todo!(),
        OpType::ReduceL1 => todo!(),
        OpType::ReduceL2 => todo!(),
        OpType::ReduceLogSum => todo!(),
        OpType::ReduceLogSumExp => todo!(),
        OpType::ReduceMax => todo!(),
        OpType::ReduceMean => todo!(),
        OpType::ReduceMin => todo!(),
        OpType::ReduceProd => todo!(),
        OpType::ReduceSum => todo!(),
        OpType::ReduceSumSquare => todo!(),
        OpType::Reshape => match inputs {
            &[data, shape] => vec![reshape::infer(data, shape)],
            _ => panic!("Reshape operator must have two inputs"),
        },
        OpType::Resize => todo!(),
        OpType::ReverseSequence => todo!(),
        OpType::RoiAlign => todo!(),
        OpType::STFT => todo!(),
        OpType::Scan => todo!(),
        OpType::Scatter => todo!(),
        OpType::ScatterElements => todo!(),
        OpType::ScatterND => todo!(),
        OpType::Selu => todo!(),
        OpType::SequenceAt => todo!(),
        OpType::SequenceConstruct => todo!(),
        OpType::SequenceEmpty => todo!(),
        OpType::SequenceErase => todo!(),
        OpType::SequenceInsert => todo!(),
        OpType::SequenceLength => todo!(),
        OpType::SequenceMap => todo!(),
        OpType::Shape => todo!(),
        OpType::Shrink => todo!(),
        OpType::Sign => todo!(),
        OpType::Size => todo!(),
        OpType::Slice => todo!(),
        OpType::Softmax => todo!(),
        OpType::SoftmaxCrossEntropyLoss => todo!(),
        OpType::Softplus => todo!(),
        OpType::Softsign => todo!(),
        OpType::SpaceToDepth => todo!(),
        OpType::Split => todo!(),
        OpType::SplitToSequence => todo!(),
        OpType::Squeeze => match inputs {
            &[data, axes] => vec![squeeze::infer(data, axes)],
            _ => panic!("Squeeze operator must have two inputs"),
        },
        OpType::StringNormalizer => todo!(),
        OpType::TfIdfVectorizer => todo!(),
        OpType::ThresholdedRelu => todo!(),
        OpType::Tile => todo!(),
        OpType::TopK => todo!(),
        OpType::Transpose => todo!(),
        OpType::Trilu => todo!(),
        OpType::Unique => todo!(),
        OpType::Unsqueeze => match inputs {
            &[data, axes] => vec![unsqueeze::infer(data, axes)],
            _ => panic!("Unsqueeze operator must have two inputs"),
        },
        OpType::Upsample => todo!(),
        OpType::Where => todo!(),
        OpType::Custom(_) => todo!(),
    }
}

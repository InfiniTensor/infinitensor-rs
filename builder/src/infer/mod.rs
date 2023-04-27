mod binary;
mod broadcast;
mod compair;
mod pool;
mod reduce;
mod reshape;
mod unary;

use crate::Tensor;
use basic_operator::OpType;

pub(crate) fn infer(op_type: OpType, inputs: &[&Tensor]) -> Vec<Tensor> {
    match op_type {
        OpType::Unary(u) => {
            assert_eq!(inputs.len(), 1, "Unary operator must have a single input");
            vec![unary::infer(u, unsafe { inputs.get_unchecked(0) })]
        }
        OpType::Binary(b) => {
            assert_eq!(inputs.len(), 2, "Binary operator must have two inputs");
            vec![binary::infer(
                b,
                unsafe { inputs.get_unchecked(0) },
                unsafe { inputs.get_unchecked(1) },
            )]
        }
        OpType::Reduce(r) => vec![reduce::infer(r, inputs)],
        OpType::Compair(_) => {
            assert_eq!(inputs.len(), 2, "Compair operator must have two inputs");
            vec![compair::infer(
                unsafe { inputs.get_unchecked(0) }, // fmt
                unsafe { inputs.get_unchecked(1) },
            )]
        }
        OpType::Pool(_) => {
            assert_eq!(inputs.len(), 5, "Pool operator must have 5 inputs");
            vec![pool::infer(
                unsafe { inputs.get_unchecked(0) },
                unsafe { inputs.get_unchecked(1) },
                unsafe { inputs.get_unchecked(2) },
                unsafe { inputs.get_unchecked(3) },
                unsafe { inputs.get_unchecked(4) },
            )]
        }
        OpType::ArgMax => todo!(),
        OpType::BatchNormalization => todo!(),
        OpType::Bernoulli => todo!(),
        OpType::BlackmanWindow => todo!(),
        OpType::Cast => todo!(),
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
        OpType::Gemm => todo!(),
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
        OpType::Identity => todo!(),
        OpType::If => todo!(),
        OpType::InstanceNormalization => todo!(),
        OpType::IsInf => todo!(),
        OpType::IsNaN => todo!(),
        OpType::LRN => todo!(),
        OpType::LSTM => todo!(),
        OpType::LayerNormalization => todo!(),
        OpType::LeakyRelu => todo!(),
        OpType::LogSoftmax => todo!(),
        OpType::Loop => todo!(),
        OpType::LpNormalization => todo!(),
        OpType::MatMul => todo!(),
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
        OpType::PRelu => todo!(),
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
        OpType::Reshape => {
            assert_eq!(inputs.len(), 2, "Reshape operator must have two inputs");
            vec![reshape::infer(
                unsafe { inputs.get_unchecked(0) }, // data
                unsafe { inputs.get_unchecked(1) }, // shape
            )]
        }
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
        OpType::Squeeze => todo!(),
        OpType::StringNormalizer => todo!(),
        OpType::TfIdfVectorizer => todo!(),
        OpType::ThresholdedRelu => todo!(),
        OpType::Tile => todo!(),
        OpType::TopK => todo!(),
        OpType::Transpose => todo!(),
        OpType::Trilu => todo!(),
        OpType::Unique => todo!(),
        OpType::Unsqueeze => todo!(),
        OpType::Upsample => todo!(),
        OpType::Where => todo!(),
        OpType::Custom(_) => todo!(),
    }
}

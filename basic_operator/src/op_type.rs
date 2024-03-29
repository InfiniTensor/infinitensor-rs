﻿use std::str::FromStr;

#[derive(Clone, Debug)]
pub enum OpType {
    Unary(Unary),
    Binary(Binary),
    Reduce(Reduce),
    Compair(Compair),
    Pool(Pool),
    ArgMax,
    BatchNormalization,
    Bernoulli,
    BlackmanWindow,
    Cast,
    CastLike,
    Celu,
    CenterCropPad,
    Clip,
    Col2lm,
    Compress,
    Concat,
    ConcatFromSequence,
    // Constant, // -> Input
    ConstantOfShape,
    Conv,
    ConvInteger,
    ConvTranspose,
    CumSum,
    DFT,
    DeformConv,
    DepthToSpace,
    DequantizeLinear,
    Det,
    Dropout,
    DynamicQuantizeLinear,
    Einsum,
    Elu,
    Expand,
    EyeLike,
    Flatten,
    GRU,
    Gather,
    GatherElements,
    GatherND,
    Gemm,
    GlobalAveragePool,
    GlobalLpPool,
    GlobalMaxPool,
    GridSample,
    GroupNormalization,
    HammingWindow,
    HannWindow,
    HardSigmoid,
    HardSwish,
    Hardmax,
    Identity,
    If,
    InstanceNormalization,
    IsInf,
    IsNaN,
    LRN,
    LSTM,
    LayerNormalization,
    LeakyRelu,
    LogSoftmax,
    Loop,
    LpNormalization,
    MatMul,
    MatMulInteger,
    MeanVarianceNormalization,
    MelWeightMatrix,
    Mish,
    Multinomial,
    NegativeLogLikelihoodLoss,
    NonMaxSuppression,
    NonZero,
    OneHot,
    Optional,
    OptionalGetElement,
    OptionalHasElement,
    PRelu,
    Pad,
    QLinearConv,
    QLinearMatMul,
    QuantizeLinear,
    RNN,
    RandomNormal,
    RandomNormalLike,
    RandomUniform,
    RandomUniformLike,
    Range,
    Reciprocal,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceProd,
    ReduceSum,
    ReduceSumSquare,
    Reshape,
    Resize,
    ReverseSequence,
    RoiAlign,
    STFT,
    Scan,
    Scatter,
    ScatterElements,
    ScatterND,
    Selu,
    SequenceAt,
    SequenceConstruct,
    SequenceEmpty,
    SequenceErase,
    SequenceInsert,
    SequenceLength,
    SequenceMap,
    Shape,
    Shrink,
    Sign,
    Size,
    Slice,
    Softmax,
    SoftmaxCrossEntropyLoss,
    Softplus,
    Softsign,
    SpaceToDepth,
    Split,
    SplitToSequence,
    Squeeze,
    StringNormalizer,
    TfIdfVectorizer,
    ThresholdedRelu,
    Tile,
    TopK,
    Transpose,
    Trilu,
    Unique,
    Unsqueeze,
    Upsample,
    Where,
    Custom(String),
}

#[derive(Clone, Copy, Debug)]
pub enum Unary {
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Ceil,
    Cos,
    Cosh,
    Erf,
    Exp,
    Floor,
    Log,
    Neg,
    Not,
    Relu,
    Round,
    Sigmoid,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
}

#[derive(Clone, Copy, Debug)]
pub enum Binary {
    Add,
    And,
    BitShift,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    Div,
    Mod,
    Mul,
    Or,
    Pow,
    Sub,
    Xor,
}

#[derive(Clone, Copy, Debug)]
pub enum Reduce {
    Max,
    Min,
    Mean,
    Sum,
}

#[derive(Clone, Copy, Debug)]
pub enum Compair {
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

#[derive(Clone, Copy, Debug)]
pub enum Pool {
    Average,
    Lp,
    Max,
    MaxRoi,
    MaxUn,
}

impl FromStr for OpType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Abs" => Ok(Self::Unary(Unary::Abs)),
            "Acos" => Ok(Self::Unary(Unary::Acos)),
            "Acosh" => Ok(Self::Unary(Unary::Acosh)),
            "Asin" => Ok(Self::Unary(Unary::Asin)),
            "Asinh" => Ok(Self::Unary(Unary::Asinh)),
            "Atan" => Ok(Self::Unary(Unary::Atan)),
            "Atanh" => Ok(Self::Unary(Unary::Atanh)),
            "Ceil" => Ok(Self::Unary(Unary::Ceil)),
            "Cos" => Ok(Self::Unary(Unary::Cos)),
            "Cosh" => Ok(Self::Unary(Unary::Cosh)),
            "Erf" => Ok(Self::Unary(Unary::Erf)),
            "Exp" => Ok(Self::Unary(Unary::Exp)),
            "Floor" => Ok(Self::Unary(Unary::Floor)),
            "Log" => Ok(Self::Unary(Unary::Log)),
            "Neg" => Ok(Self::Unary(Unary::Neg)),
            "Not" => Ok(Self::Unary(Unary::Not)),
            "Relu" => Ok(Self::Unary(Unary::Relu)),
            "Round" => Ok(Self::Unary(Unary::Round)),
            "Sigmoid" => Ok(Self::Unary(Unary::Sigmoid)),
            "Sin" => Ok(Self::Unary(Unary::Sin)),
            "Sinh" => Ok(Self::Unary(Unary::Sinh)),
            "Sqrt" => Ok(Self::Unary(Unary::Sqrt)),
            "Tan" => Ok(Self::Unary(Unary::Tan)),
            "Tanh" => Ok(Self::Unary(Unary::Tanh)),

            "Add" => Ok(Self::Binary(Binary::Add)),
            "And" => Ok(Self::Binary(Binary::And)),
            "BitShift" => Ok(Self::Binary(Binary::BitShift)),
            "BitwiseAnd" => Ok(Self::Binary(Binary::BitwiseAnd)),
            "BitwiseNot" => Ok(Self::Binary(Binary::BitwiseNot)),
            "BitwiseOr" => Ok(Self::Binary(Binary::BitwiseOr)),
            "BitwiseXor" => Ok(Self::Binary(Binary::BitwiseXor)),
            "Div" => Ok(Self::Binary(Binary::Div)),
            "Mod" => Ok(Self::Binary(Binary::Mod)),
            "Mul" => Ok(Self::Binary(Binary::Mul)),
            "Or" => Ok(Self::Binary(Binary::Or)),
            "Pow" => Ok(Self::Binary(Binary::Pow)),
            "Sub" => Ok(Self::Binary(Binary::Sub)),
            "Xor" => Ok(Self::Binary(Binary::Xor)),

            "Max" => Ok(Self::Reduce(Reduce::Max)),
            "Min" => Ok(Self::Reduce(Reduce::Min)),
            "Mean" => Ok(Self::Reduce(Reduce::Mean)),
            "Sum" => Ok(Self::Reduce(Reduce::Sum)),

            "Equal" => Ok(Self::Compair(Compair::Equal)),
            "Greater" => Ok(Self::Compair(Compair::Greater)),
            "GreaterOrEqual" => Ok(Self::Compair(Compair::GreaterOrEqual)),
            "Less" => Ok(Self::Compair(Compair::Less)),
            "LessOrEqual" => Ok(Self::Compair(Compair::LessOrEqual)),

            "AveragePool" => Ok(Self::Pool(Pool::Average)),
            "LpPool" => Ok(Self::Pool(Pool::Lp)),
            "MaxPool" => Ok(Self::Pool(Pool::Max)),
            "MaxRoiPool" => Ok(Self::Pool(Pool::MaxRoi)),
            "MaxUnPool" => Ok(Self::Pool(Pool::MaxUn)),

            "ArgMax" => Ok(Self::ArgMax),
            "BatchNormalization" => Ok(Self::BatchNormalization),
            "Bernoulli" => Ok(Self::Bernoulli),
            "BlackmanWindow" => Ok(Self::BlackmanWindow),
            "Cast" => Ok(Self::Cast),
            "CastLike" => Ok(Self::CastLike),
            "Celu" => Ok(Self::Celu),
            "CenterCropPad" => Ok(Self::CenterCropPad),
            "Clip" => Ok(Self::Clip),
            "Col2lm" => Ok(Self::Col2lm),
            "Compress" => Ok(Self::Compress),
            "Concat" => Ok(Self::Concat),
            "ConcatFromSequence" => Ok(Self::ConcatFromSequence),
            "ConstantOfShape" => Ok(Self::ConstantOfShape),
            "Conv" => Ok(Self::Conv),
            "ConvInteger" => Ok(Self::ConvInteger),
            "ConvTranspose" => Ok(Self::ConvTranspose),
            "CumSum" => Ok(Self::CumSum),
            "DFT" => Ok(Self::DFT),
            "DeformConv" => Ok(Self::DeformConv),
            "DepthToSpace" => Ok(Self::DepthToSpace),
            "DequantizeLinear" => Ok(Self::DequantizeLinear),
            "Det" => Ok(Self::Det),
            "Dropout" => Ok(Self::Dropout),
            "DynamicQuantizeLinear" => Ok(Self::DynamicQuantizeLinear),
            "Einsum" => Ok(Self::Einsum),
            "Elu" => Ok(Self::Elu),
            "Expand" => Ok(Self::Expand),
            "EyeLike" => Ok(Self::EyeLike),
            "Flatten" => Ok(Self::Flatten),
            "GRU" => Ok(Self::GRU),
            "Gather" => Ok(Self::Gather),
            "GatherElements" => Ok(Self::GatherElements),
            "GatherND" => Ok(Self::GatherND),
            "Gemm" => Ok(Self::Gemm),
            "GlobalAveragePool" => Ok(Self::GlobalAveragePool),
            "GlobalLpPool" => Ok(Self::GlobalLpPool),
            "GlobalMaxPool" => Ok(Self::GlobalMaxPool),
            "GridSample" => Ok(Self::GridSample),
            "GroupNormalization" => Ok(Self::GroupNormalization),
            "HammingWindow" => Ok(Self::HammingWindow),
            "HannWindow" => Ok(Self::HannWindow),
            "HardSigmoid" => Ok(Self::HardSigmoid),
            "HardSwish" => Ok(Self::HardSwish),
            "Hardmax" => Ok(Self::Hardmax),
            "Identity" => Ok(Self::Identity),
            "If" => Ok(Self::If),
            "InstanceNormalization" => Ok(Self::InstanceNormalization),
            "IsInf" => Ok(Self::IsInf),
            "IsNaN" => Ok(Self::IsNaN),
            "LRN" => Ok(Self::LRN),
            "LSTM" => Ok(Self::LSTM),
            "LayerNormalization" => Ok(Self::LayerNormalization),
            "LeakyRelu" => Ok(Self::LeakyRelu),
            "LogSoftmax" => Ok(Self::LogSoftmax),
            "Loop" => Ok(Self::Loop),
            "LpNormalization" => Ok(Self::LpNormalization),
            "MatMul" => Ok(Self::MatMul),
            "MatMulInteger" => Ok(Self::MatMulInteger),
            "MeanVarianceNormalization" => Ok(Self::MeanVarianceNormalization),
            "MelWeightMatrix" => Ok(Self::MelWeightMatrix),
            "Mish" => Ok(Self::Mish),
            "Multinomial" => Ok(Self::Multinomial),
            "NegativeLogLikelihoodLoss" => Ok(Self::NegativeLogLikelihoodLoss),
            "NonMaxSuppression" => Ok(Self::NonMaxSuppression),
            "NonZero" => Ok(Self::NonZero),
            "OneHot" => Ok(Self::OneHot),
            "Optional" => Ok(Self::Optional),
            "OptionalGetElement" => Ok(Self::OptionalGetElement),
            "OptionalHasElement" => Ok(Self::OptionalHasElement),
            "PRelu" => Ok(Self::PRelu),
            "Pad" => Ok(Self::Pad),
            "QLinearConv" => Ok(Self::QLinearConv),
            "QLinearMatMul" => Ok(Self::QLinearMatMul),
            "QuantizeLinear" => Ok(Self::QuantizeLinear),
            "RNN" => Ok(Self::RNN),
            "RandomNormal" => Ok(Self::RandomNormal),
            "RandomNormalLike" => Ok(Self::RandomNormalLike),
            "RandomUniform" => Ok(Self::RandomUniform),
            "RandomUniformLike" => Ok(Self::RandomUniformLike),
            "Range" => Ok(Self::Range),
            "Reciprocal" => Ok(Self::Reciprocal),
            "ReduceL1" => Ok(Self::ReduceL1),
            "ReduceL2" => Ok(Self::ReduceL2),
            "ReduceLogSum" => Ok(Self::ReduceLogSum),
            "ReduceLogSumExp" => Ok(Self::ReduceLogSumExp),
            "ReduceMax" => Ok(Self::ReduceMax),
            "ReduceMean" => Ok(Self::ReduceMean),
            "ReduceMin" => Ok(Self::ReduceMin),
            "ReduceProd" => Ok(Self::ReduceProd),
            "ReduceSum" => Ok(Self::ReduceSum),
            "ReduceSumSquare" => Ok(Self::ReduceSumSquare),
            "Reshape" => Ok(Self::Reshape),
            "Resize" => Ok(Self::Resize),
            "ReverseSequence" => Ok(Self::ReverseSequence),
            "RoiAlign" => Ok(Self::RoiAlign),
            "STFT" => Ok(Self::STFT),
            "Scan" => Ok(Self::Scan),
            "Scatter" => Ok(Self::Scatter),
            "ScatterElements" => Ok(Self::ScatterElements),
            "ScatterND" => Ok(Self::ScatterND),
            "Selu" => Ok(Self::Selu),
            "SequenceAt" => Ok(Self::SequenceAt),
            "SequenceConstruct" => Ok(Self::SequenceConstruct),
            "SequenceEmpty" => Ok(Self::SequenceEmpty),
            "SequenceErase" => Ok(Self::SequenceErase),
            "SequenceInsert" => Ok(Self::SequenceInsert),
            "SequenceLength" => Ok(Self::SequenceLength),
            "SequenceMap" => Ok(Self::SequenceMap),
            "Shape" => Ok(Self::Shape),
            "Shrink" => Ok(Self::Shrink),
            "Sign" => Ok(Self::Sign),
            "Size" => Ok(Self::Size),
            "Slice" => Ok(Self::Slice),
            "Softmax" => Ok(Self::Softmax),
            "SoftmaxCrossEntropyLoss" => Ok(Self::SoftmaxCrossEntropyLoss),
            "Softplus" => Ok(Self::Softplus),
            "Softsign" => Ok(Self::Softsign),
            "SpaceToDepth" => Ok(Self::SpaceToDepth),
            "Split" => Ok(Self::Split),
            "SplitToSequence" => Ok(Self::SplitToSequence),
            "Squeeze" => Ok(Self::Squeeze),
            "StringNormalizer" => Ok(Self::StringNormalizer),
            "TfIdfVectorizer" => Ok(Self::TfIdfVectorizer),
            "ThresholdedRelu" => Ok(Self::ThresholdedRelu),
            "Tile" => Ok(Self::Tile),
            "TopK" => Ok(Self::TopK),
            "Transpose" => Ok(Self::Transpose),
            "Trilu" => Ok(Self::Trilu),
            "Unique" => Ok(Self::Unique),
            "Unsqueeze" => Ok(Self::Unsqueeze),
            "Upsample" => Ok(Self::Upsample),
            "Where" => Ok(Self::Where),

            _ => Err(()),
        }
    }
}

﻿#[derive(Clone, Debug)]
pub enum OpType {
    Unary(Unary),
    Binary(Binary),
    Reduce(Reduce),
    Compair(Compair),
    ArgMax,
    AveragePool,
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
    LpPool,
    MatMul,
    MatMulInteger,
    MaxPool,
    MaxRoiPool,
    MaxUnpool,
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

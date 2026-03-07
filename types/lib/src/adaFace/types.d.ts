import * as tf from '@tensorflow/tfjs';
/**
 * Convolution + BatchNorm parameters.
 */
export interface ConvBnParams {
    weights: tf.Tensor4D;
    stride?: number;
    bn_mean: tf.Tensor1D;
    bn_variance: tf.Tensor1D;
    bn_offset: tf.Tensor1D;
    bn_scale: tf.Tensor1D;
}
/**
 * Squeeze-Excitation parameters.
 */
export interface SEParams {
    fc1: tf.Tensor2D;
    fc2: tf.Tensor2D;
}
/**
 * IR-SE block parameters.
 */
export interface IRSEBlockParams {
    depthwise: tf.Tensor4D;
    bn1_mean: tf.Tensor1D;
    bn1_variance: tf.Tensor1D;
    bn1_offset: tf.Tensor1D;
    bn1_scale: tf.Tensor1D;
    se: SEParams;
    pointwise: tf.Tensor4D;
    bn2_mean: tf.Tensor1D;
    bn2_variance: tf.Tensor1D;
    bn2_offset: tf.Tensor1D;
    bn2_scale: tf.Tensor1D;
}
/**
 * Stem parameters.
 */
export interface StemParams {
    conv1: ConvBnParams;
    conv2: ConvBnParams;
    conv3: ConvBnParams;
}
/**
 * Fully connected layer parameters.
 */
export interface FCParams {
    weights: tf.Tensor2D;
    bias: tf.Tensor1D;
}
/**
 * AdaFace network parameters.
 */
export interface NetParams {
    stem: StemParams;
    stage1: IRSEBlockParams;
    stage2: IRSEBlockParams;
    stage3: IRSEBlockParams;
    stage4: IRSEBlockParams;
    fc: FCParams;
}

import * as tf from '../../dist/tfjs.esm';

/**
 * Convolution block parameters.
 */
export interface ConvBlockParams {
  weights: tf.Tensor4D;
  bn_mean: tf.Tensor1D;
  bn_variance: tf.Tensor1D;
  bn_offset: tf.Tensor1D;
  bn_scale: tf.Tensor1D;
}

/**
 * Residual block parameters.
 */
export interface ResidualBlockParams {
  conv1: ConvBlockParams;
  conv2: ConvBlockParams;
}

/**
 * Encoder parameters.
 */
export interface EncoderParams {
  conv1: ConvBlockParams;
  conv2: ConvBlockParams;
  conv3: ConvBlockParams;
  conv4: ConvBlockParams;
  conv5: ConvBlockParams;
}

/**
 * Decoder parameters.
 */
export interface DecoderParams {
  conv1: ConvBlockParams;
}

/**
 * Landmark head parameters.
 */
export interface LandmarkHeadParams {
  weights: tf.Tensor4D;
  bias: tf.Tensor1D;
}

/**
 * FaceMesh network parameters.
 */
export interface NetParams {
  encoder: EncoderParams;
  bottleneck: ResidualBlockParams[];
  decoder: DecoderParams;
  landmarkHead: LandmarkHeadParams;
}

import * as tf from '../../dist/tfjs.esm';

/**
 * FaceMesh convolution block parameters.
 */
export interface FaceMeshConvBlockParams {
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
  conv1: FaceMeshConvBlockParams;
  conv2: FaceMeshConvBlockParams;
}

/**
 * Encoder parameters.
 */
export interface EncoderParams {
  conv1: FaceMeshConvBlockParams;
  conv2: FaceMeshConvBlockParams;
  conv3: FaceMeshConvBlockParams;
  conv4: FaceMeshConvBlockParams;
  conv5: FaceMeshConvBlockParams;
}

/**
 * Decoder parameters.
 */
export interface DecoderParams {
  conv1: FaceMeshConvBlockParams;
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

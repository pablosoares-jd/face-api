import type * as tf from '@tensorflow/tfjs';

/**
 * BlazeFace convolution block parameters.
 */
export interface ConvBlockParams {
  depthwise: tf.Tensor4D;
  depthwiseBias: tf.Tensor1D;
  pointwise: tf.Tensor4D;
  pointwiseBias: tf.Tensor1D;
}

/**
 * BlazeFace detection head parameters.
 */
export interface HeadParams {
  weights: tf.Tensor4D;
  bias: tf.Tensor1D;
}

/**
 * BlazeFace network parameters.
 */
export interface NetParams {
  conv1: ConvBlockParams;
  conv2: ConvBlockParams;
  conv3: ConvBlockParams;
  conv4: ConvBlockParams;
  conv5: ConvBlockParams;
  conv6: ConvBlockParams;
  conv7: ConvBlockParams;
  conv8: ConvBlockParams;
  classifierHead: HeadParams;
  regressorHead: HeadParams;
}

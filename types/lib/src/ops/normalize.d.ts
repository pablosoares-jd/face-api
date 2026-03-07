import * as tf from '@tensorflow/tfjs';
/**
 * Normalize an image tensor by subtracting mean RGB values.
 * Optimized to use a single tensor operation with broadcasting.
 *
 * @param x Input tensor of shape [batch, height, width, 3]
 * @param meanRgb Array of [R, G, B] mean values to subtract
 * @returns Normalized tensor with same shape as input
 */
export declare function normalize(x: tf.Tensor4D, meanRgb: number[]): tf.Tensor4D;

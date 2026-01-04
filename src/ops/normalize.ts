import * as tf from '../../dist/tfjs.esm';

/**
 * Normalize an image tensor by subtracting mean RGB values.
 * Optimized to use a single tensor operation with broadcasting.
 *
 * @param x Input tensor of shape [batch, height, width, 3]
 * @param meanRgb Array of [R, G, B] mean values to subtract
 * @returns Normalized tensor with same shape as input
 */
export function normalize(x: tf.Tensor4D, meanRgb: number[]): tf.Tensor4D {
  return tf.tidy(() => {
    // Create a single [1, 1, 1, 3] tensor that broadcasts across the image
    // This is more efficient than creating 3 separate fill tensors and concatenating
    const meanTensor = tf.tensor1d(meanRgb, 'float32').reshape([1, 1, 1, 3]);

    // Broadcasting handles the rest - no need for tf.fill or tf.concat
    return tf.sub(x, meanTensor) as tf.Tensor4D;
  });
}

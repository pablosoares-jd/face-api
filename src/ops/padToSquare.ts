import * as tf from '@tensorflow/tfjs';

/**
 * Pads the smaller dimension of an image tensor with zeros, such that width === height.
 * Uses tf.pad() for better performance instead of creating separate zero tensors.
 *
 * @param imgTensor The image tensor.
 * @param isCenterImage (optional, default: false) If true, add an equal amount of padding on
 * both sides of the minor dimension of the image.
 * @returns The padded tensor with width === height.
 */
export function padToSquare(imgTensor: tf.Tensor4D, isCenterImage = false): tf.Tensor4D {
  return tf.tidy(() => {
    const [height, width] = imgTensor.shape.slice(1);

    // No padding needed if already square
    if (height === width) return imgTensor;

    const dimDiff = Math.abs(height - width);
    const isHeightLarger = height > width;

    // Calculate padding amounts for before and after
    let padBefore: number;
    let padAfter: number;

    if (isCenterImage) {
      padBefore = Math.floor(dimDiff / 2);
      padAfter = dimDiff - padBefore;
    } else {
      padBefore = 0;
      padAfter = dimDiff;
    }

    // Build padding configuration: [batch, height, width, channels]
    // tf.pad expects [[beforeBatch, afterBatch], [beforeH, afterH], [beforeW, afterW], [beforeC, afterC]]
    const paddings: [number, number][] = isHeightLarger
      ? [[0, 0], [0, 0], [padBefore, padAfter], [0, 0]]  // Pad width
      : [[0, 0], [padBefore, padAfter], [0, 0], [0, 0]]; // Pad height

    return tf.pad(imgTensor, paddings) as tf.Tensor4D;
  });
}

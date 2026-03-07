import type * as tf from '@tensorflow/tfjs';
/**
 * Non-Maximum Suppression for SSD MobileNet.
 * Uses pre-fetched box data to avoid GPU blocking during IOU calculations.
 *
 * @deprecated Use `nonMaxSuppressionFast` from '../ops/nonMaxSuppression' instead.
 * This function is kept for backward compatibility.
 *
 * Note: The boxes tensor data should be fetched before calling this function
 * using async boxes.array() for better performance.
 */
export declare function nonMaxSuppression(boxes: tf.Tensor2D | number[][], scores: number[], maxOutputSize: number, iouThreshold: number, scoreThreshold: number): number[];

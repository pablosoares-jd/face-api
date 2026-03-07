import type * as tf from '@tensorflow/tfjs';

import { nonMaxSuppressionFast } from '../ops/nonMaxSuppression';

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
export function nonMaxSuppression(
  boxes: tf.Tensor2D | number[][],
  scores: number[],
  maxOutputSize: number,
  iouThreshold: number,
  scoreThreshold: number,
): number[] {
  // If boxes is a tensor, extract data synchronously (caller should pre-fetch for perf)
  // This maintains backward compatibility while allowing optimized usage
  const boxesData: number[][] = Array.isArray(boxes)
    ? boxes
    : (boxes as tf.Tensor2D).arraySync() as number[][];

  return nonMaxSuppressionFast(boxesData, scores, maxOutputSize, iouThreshold, scoreThreshold);
}

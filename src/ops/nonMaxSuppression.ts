import { Box } from '../classes/Box';
import type { BoxArray } from './iou';
import { iou, iouFromArrays } from './iou';

/**
 * Options for Non-Maximum Suppression.
 */
export interface NMSOptions {
  /** Maximum number of boxes to return. Default: Infinity */
  maxResults?: number;
  /** Minimum score threshold. Boxes with scores below this are filtered. Default: 0 */
  scoreThreshold?: number;
  /** Use standard IOU (true) or intersection over minimum area (false). Default: true */
  isIOU?: boolean;
}

/**
 * Unified Non-Maximum Suppression implementation.
 * Supports both Box objects and array-based boxes for flexibility and performance.
 *
 * @param boxes Array of boxes (either Box objects or [top, left, bottom, right] arrays)
 * @param scores Array of confidence scores for each box
 * @param iouThreshold IOU threshold for suppression (boxes with IOU > threshold are suppressed)
 * @param options Additional NMS options
 * @returns Indices of selected boxes sorted by score (highest first)
 *
 * @example
 * // With Box objects
 * const indices = nonMaxSuppression(boxObjects, scores, 0.5);
 *
 * @example
 * // With array boxes
 * const boxes = [[0.1, 0.1, 0.5, 0.5], [0.15, 0.15, 0.55, 0.55]];
 * const indices = nonMaxSuppression(boxes, scores, 0.5, { maxResults: 10 });
 */
export function nonMaxSuppression(
  boxes: Box[] | BoxArray[],
  scores: number[],
  iouThreshold: number,
  options: NMSOptions | boolean = {},
): number[] {
  // Handle legacy signature where 4th param was isIOU boolean
  const opts: NMSOptions = typeof options === 'boolean'
    ? { isIOU: options }
    : options;

  const {
    maxResults = Infinity,
    scoreThreshold = 0,
    isIOU = true,
  } = opts;

  // Determine if boxes are Box objects or arrays
  const isBoxObjects = boxes.length > 0 && boxes[0] instanceof Box;

  // Filter by score threshold and sort by score descending
  const candidates = scores
    .map((score, boxIndex) => ({ score, boxIndex }))
    .filter((c) => c.score >= scoreThreshold)
    .sort((c1, c2) => c2.score - c1.score);

  const selected: number[] = [];

  for (const candidate of candidates) {
    if (selected.length >= maxResults) break;

    const candidateBox = boxes[candidate.boxIndex];
    if (!candidateBox) continue;

    // Check IOU against all previously selected boxes
    let dominated = false;
    for (const selectedIdx of selected) {
      const selectedBox = boxes[selectedIdx];
      if (!selectedBox) continue;

      let iouValue: number;
      if (isBoxObjects) {
        iouValue = iou(candidateBox as Box, selectedBox as Box, isIOU);
      } else {
        iouValue = iouFromArrays(candidateBox as BoxArray, selectedBox as BoxArray);
      }

      if (iouValue > iouThreshold) {
        dominated = true;
        break;
      }
    }

    if (!dominated) {
      selected.push(candidate.boxIndex);
    }
  }

  return selected;
}

/**
 * Non-Maximum Suppression optimized for 2D array box data.
 * This variant is optimized for use with pre-fetched tensor data.
 *
 * @param boxesData 2D array of boxes where each box is [top, left, bottom, right]
 * @param scores Array of confidence scores
 * @param maxResults Maximum number of boxes to return
 * @param iouThreshold IOU threshold for suppression
 * @param scoreThreshold Minimum score threshold
 * @returns Indices of selected boxes
 */
export function nonMaxSuppressionFast(
  boxesData: number[][],
  scores: number[],
  maxResults: number,
  iouThreshold: number,
  scoreThreshold: number,
): number[] {
  return nonMaxSuppression(
    boxesData,
    scores,
    iouThreshold,
    { maxResults, scoreThreshold, isIOU: true },
  );
}

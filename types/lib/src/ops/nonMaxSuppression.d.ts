import { Box } from '../classes/Box';
import type { BoxArray } from './iou';
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
export declare function nonMaxSuppression(boxes: Box[] | BoxArray[], scores: number[], iouThreshold: number, options?: NMSOptions | boolean): number[];
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
export declare function nonMaxSuppressionFast(boxesData: number[][], scores: number[], maxResults: number, iouThreshold: number, scoreThreshold: number): number[];

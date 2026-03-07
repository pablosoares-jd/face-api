import type { Box } from '../classes/Box';
/**
 * Box format for array-based IOU calculation.
 * Format: [top, left, bottom, right] or [y1, x1, y2, x2]
 */
export type BoxArray = [number, number, number, number] | number[];
/**
 * Calculate IOU (Intersection over Union) between two Box objects.
 * @param box1 First box
 * @param box2 Second box
 * @param isIOU If true, calculates standard IOU. If false, calculates intersection over minimum area.
 * @returns IOU value between 0 and 1
 */
export declare function iou(box1: Box, box2: Box, isIOU?: boolean): number;
/**
 * Calculate IOU between two boxes in array format [top, left, bottom, right].
 * Optimized version that avoids Box object creation.
 * @param boxA First box as [top, left, bottom, right]
 * @param boxB Second box as [top, left, bottom, right]
 * @returns IOU value between 0 and 1
 */
export declare function iouFromArrays(boxA: BoxArray, boxB: BoxArray): number;
/**
 * Calculate IOU from box data arrays at specific indices.
 * Useful for NMS when boxes are stored in a 2D array.
 * @param boxesData Array of boxes
 * @param i Index of first box
 * @param j Index of second box
 * @returns IOU value between 0 and 1
 */
export declare function iouAtIndices(boxesData: number[][], i: number, j: number): number;

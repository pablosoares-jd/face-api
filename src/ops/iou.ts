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
export function iou(box1: Box, box2: Box, isIOU = true): number {
  const width = Math.max(0.0, Math.min(box1.right, box2.right) - Math.max(box1.left, box2.left));
  const height = Math.max(0.0, Math.min(box1.bottom, box2.bottom) - Math.max(box1.top, box2.top));
  const interSection = width * height;

  return isIOU
    ? interSection / (box1.area + box2.area - interSection)
    : interSection / Math.min(box1.area, box2.area);
}

/**
 * Calculate IOU between two boxes in array format [top, left, bottom, right].
 * Optimized version that avoids Box object creation.
 * @param boxA First box as [top, left, bottom, right]
 * @param boxB Second box as [top, left, bottom, right]
 * @returns IOU value between 0 and 1
 */
export function iouFromArrays(boxA: BoxArray, boxB: BoxArray): number {
  const topA = boxA[0] ?? 0;
  const leftA = boxA[1] ?? 0;
  const bottomA = boxA[2] ?? 0;
  const rightA = boxA[3] ?? 0;

  const topB = boxB[0] ?? 0;
  const leftB = boxB[1] ?? 0;
  const bottomB = boxB[2] ?? 0;
  const rightB = boxB[3] ?? 0;

  // Calculate areas
  const areaA = (bottomA - topA) * (rightA - leftA);
  const areaB = (bottomB - topB) * (rightB - leftB);

  if (areaA <= 0 || areaB <= 0) return 0;

  // Calculate intersection
  const intersectTop = Math.max(topA, topB);
  const intersectLeft = Math.max(leftA, leftB);
  const intersectBottom = Math.min(bottomA, bottomB);
  const intersectRight = Math.min(rightA, rightB);

  const intersectWidth = Math.max(0, intersectRight - intersectLeft);
  const intersectHeight = Math.max(0, intersectBottom - intersectTop);
  const intersectArea = intersectWidth * intersectHeight;

  // Calculate union
  const unionArea = areaA + areaB - intersectArea;

  return unionArea > 0 ? intersectArea / unionArea : 0;
}

/**
 * Calculate IOU from box data arrays at specific indices.
 * Useful for NMS when boxes are stored in a 2D array.
 * @param boxesData Array of boxes
 * @param i Index of first box
 * @param j Index of second box
 * @returns IOU value between 0 and 1
 */
export function iouAtIndices(boxesData: number[][], i: number, j: number): number {
  const boxI = boxesData[i];
  const boxJ = boxesData[j];

  if (!boxI || !boxJ) return 0;

  return iouFromArrays(boxI, boxJ);
}

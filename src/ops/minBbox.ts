import type { IPoint } from '../classes/index';
import { BoundingBox } from '../classes/index';

/**
 * Calculate minimum bounding box for a set of points.
 * @param pts Array of points
 * @returns BoundingBox containing all points
 */
export function minBbox(pts: IPoint[]): BoundingBox {
  // Filter out invalid points (undefined, NaN, Infinity coordinates)
  const validPts = pts.filter((pt) => pt
    && typeof pt.x === 'number' && Number.isFinite(pt.x)
    && typeof pt.y === 'number' && Number.isFinite(pt.y));

  // Handle empty or all-invalid points case
  if (validPts.length === 0) {
    console.warn('minBbox: No valid points provided, returning zero-area box');
    return new BoundingBox(0, 0, 0, 0, true);
  }

  const xs = validPts.map((pt) => pt.x);
  const ys = validPts.map((pt) => pt.y);

  // Use first point as initial value to avoid Infinity
  const firstX = xs[0] ?? 0;
  const firstY = ys[0] ?? 0;

  const minX = xs.reduce((min, x) => (x < min ? x : min), firstX);
  const minY = ys.reduce((min, y) => (y < min ? y : min), firstY);
  const maxX = xs.reduce((max, x) => (max < x ? x : max), firstX);
  const maxY = ys.reduce((max, y) => (max < y ? y : max), firstY);

  return new BoundingBox(minX, minY, maxX, maxY);
}

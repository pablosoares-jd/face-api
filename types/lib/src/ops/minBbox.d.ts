import type { IPoint } from '../classes/index';
import { BoundingBox } from '../classes/index';
/**
 * Calculate minimum bounding box for a set of points.
 * @param pts Array of points
 * @returns BoundingBox containing all points
 */
export declare function minBbox(pts: IPoint[]): BoundingBox;

import type { IDimensions } from '../classes/Dimensions';
import { Dimensions } from '../classes/Dimensions';
/**
 * Get the dimensions of a media element or dimensions object.
 *
 * @param input Media element or object with width/height properties
 * @returns Dimensions object with width and height
 * @throws Error if dimensions are invalid (zero or negative)
 */
export declare function getMediaDimensions(input: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | IDimensions): Dimensions;

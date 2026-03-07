import type { IDimensions } from '../classes/Dimensions';
import { Dimensions } from '../classes/Dimensions';
import { env } from '../env/index';

/**
 * Get the dimensions of a media element or dimensions object.
 *
 * @param input Media element or object with width/height properties
 * @returns Dimensions object with width and height
 * @throws Error if dimensions are invalid (zero or negative)
 */
export function getMediaDimensions(input: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | IDimensions): Dimensions {
  const { Image, Video } = env.getEnv();

  let width: number;
  let height: number;

  if (input instanceof Image) {
    width = input.naturalWidth;
    height = input.naturalHeight;

    if (width === 0 || height === 0) {
      throw new Error(
        'getMediaDimensions - image has zero dimensions. '
        + 'Make sure the image is fully loaded before processing. '
        + `Current dimensions: ${width}x${height}`,
      );
    }
  } else if (input instanceof Video) {
    width = input.videoWidth;
    height = input.videoHeight;

    if (width === 0 || height === 0) {
      throw new Error(
        'getMediaDimensions - video has zero dimensions. '
        + 'Make sure the video metadata is loaded (readyState >= 1) before processing. '
        + `Current dimensions: ${width}x${height}, readyState: ${input.readyState}`,
      );
    }
  } else {
    width = input.width;
    height = input.height;
  }

  // Validate dimensions
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new Error(`getMediaDimensions - invalid dimensions: width=${width}, height=${height}`);
  }

  if (width <= 0 || height <= 0) {
    throw new Error(`getMediaDimensions - dimensions must be positive: width=${width}, height=${height}`);
  }

  return new Dimensions(width, height);
}

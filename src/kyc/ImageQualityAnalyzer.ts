import * as tf from '@tensorflow/tfjs';

import type { FaceDetection } from '../classes/FaceDetection';
import type { TNetInput } from '../dom/index';
import { toNetInput } from '../dom/index';

/**
 * Image quality metrics result.
 */
export interface ImageQualityResult {
  /** Sharpness score (0-1, higher = sharper) */
  sharpness: number;

  /** Brightness score (0-1, 0.5 = optimal) */
  brightness: number;

  /** Contrast score (0-1, higher = more contrast) */
  contrast: number;

  /** Whether the face region was successfully extracted */
  faceRegionExtracted: boolean;
}

/**
 * Laplacian kernel for edge detection (sharpness estimation).
 * This 3x3 kernel approximates the second derivative of the image.
 */
const LAPLACIAN_KERNEL = [
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0],
];

/**
 * Image Quality Analyzer using real pixel analysis.
 *
 * Implements:
 * - Laplacian variance for sharpness/blur detection
 * - Mean pixel intensity for brightness
 * - Standard deviation for contrast
 *
 * All tensor operations use async data() instead of blocking dataSync()
 * for better GPU pipeline efficiency.
 */
export class ImageQualityAnalyzer {
  /**
   * Analyze image quality for the face region.
   */
  public async analyze(
    input: TNetInput,
    detection: FaceDetection,
  ): Promise<ImageQualityResult> {
    const netInput = await toNetInput(input);

    // Get input tensor
    const inputTensor = netInput.getInput(0);
    if (!(inputTensor instanceof tf.Tensor)) {
      return this.getDefaultResult();
    }

    // Create tensors that need async data extraction
    // Convert to grayscale for quality analysis
    const grayscale = tf.tidy(() => this.toGrayscale(inputTensor as tf.Tensor3D));

    // Extract face region
    const faceRegion = this.extractFaceRegion(grayscale, detection, netInput.getInputWidth(0), netInput.getInputHeight(0));
    grayscale.dispose();

    if (!faceRegion) {
      return this.getDefaultResult();
    }

    // Calculate metric tensors (keep them for async extraction)
    const sharpnessTensor = this.calculateSharpnessTensor(faceRegion);
    const brightnessTensor = faceRegion.mean();
    const contrastTensor = this.calculateContrastTensor(faceRegion);
    faceRegion.dispose();

    try {
      // Extract values asynchronously (non-blocking)
      const [sharpnessData, brightnessData, contrastData] = await Promise.all([
        sharpnessTensor.data(),
        brightnessTensor.data(),
        contrastTensor.data(),
      ]);

      // Process values
      const variance = sharpnessData[0] ?? 0;
      const sharpness = Math.min(1, Math.max(0, variance * 100));

      const meanBrightness = brightnessData[0] ?? 0.5;
      const brightness = Math.min(1, Math.max(0, meanBrightness));

      const contrastVariance = contrastData[0] ?? 0;
      const stdDev = Math.sqrt(contrastVariance);
      const contrast = Math.min(1, stdDev * 4);

      return {
        sharpness,
        brightness,
        contrast,
        faceRegionExtracted: true,
      };
    } finally {
      // Dispose tensors
      sharpnessTensor.dispose();
      brightnessTensor.dispose();
      contrastTensor.dispose();
    }
  }

  /**
   * Convert RGB tensor to grayscale.
   * Assumes input is in [0, 255] range and normalizes to [0, 1].
   */
  private toGrayscale(tensor: tf.Tensor3D): tf.Tensor2D {
    return tf.tidy(() => {
      // Always normalize to 0-1 (assume input is 0-255 for consistency)
      // This avoids the blocking dataSync() call
      const normalized = tf.div(tensor, 255) as tf.Tensor3D;

      // RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
      const channels = tf.split(normalized, 3, 2);
      const r = channels[0];
      const g = channels[1];
      const b = channels[2];

      if (!r || !g || !b) {
        throw new Error('ImageQualityAnalyzer.toGrayscale - expected 3 color channels');
      }

      const gray = tf.add(
        tf.add(
          tf.mul(r, 0.299),
          tf.mul(g, 0.587),
        ),
        tf.mul(b, 0.114),
      );

      return gray.squeeze([2]) as tf.Tensor2D;
    });
  }

  /**
   * Extract face region from grayscale image.
   */
  private extractFaceRegion(
    grayscale: tf.Tensor2D,
    detection: FaceDetection,
    imageWidth: number,
    imageHeight: number,
  ): tf.Tensor2D | null {
    const box = detection.box;

    // Calculate pixel coordinates with padding
    const padding = 0.1;
    const x1 = Math.max(0, Math.floor((box.x - box.width * padding) * imageWidth / imageWidth));
    const y1 = Math.max(0, Math.floor((box.y - box.height * padding) * imageHeight / imageHeight));
    const x2 = Math.min(imageWidth, Math.ceil((box.x + box.width * (1 + padding)) * imageWidth / imageWidth));
    const y2 = Math.min(imageHeight, Math.ceil((box.y + box.height * (1 + padding)) * imageHeight / imageHeight));

    // Ensure valid region
    const width = x2 - x1;
    const height = y2 - y1;

    if (width <= 10 || height <= 10) {
      return null;
    }

    // Slice face region
    try {
      return tf.tidy(() => {
        const sliced = tf.slice(grayscale, [y1, x1], [height, width]);
        return sliced as tf.Tensor2D;
      });
    } catch {
      return null;
    }
  }

  /**
   * Calculate sharpness tensor using Laplacian variance.
   * Returns variance tensor for async extraction.
   * Higher variance = sharper image.
   */
  private calculateSharpnessTensor(faceRegion: tf.Tensor2D): tf.Tensor {
    // Create Laplacian kernel tensor
    const kernel = tf.tensor4d(
      LAPLACIAN_KERNEL.flat(),
      [3, 3, 1, 1],
    );

    // Reshape for convolution: [batch, height, width, channels]
    const input4d = faceRegion.expandDims(0).expandDims(-1) as tf.Tensor4D;

    // Apply Laplacian convolution
    const laplacian = tf.conv2d(input4d, kernel, 1, 'same');

    // Calculate variance of Laplacian response
    const mean = laplacian.mean();
    const squaredDiff = tf.square(tf.sub(laplacian, mean));
    const variance = squaredDiff.mean();

    // Clean up intermediate tensors
    kernel.dispose();
    input4d.dispose();
    laplacian.dispose();
    mean.dispose();
    squaredDiff.dispose();

    return variance;
  }

  /**
   * Calculate contrast tensor as variance of pixel intensities.
   * Returns variance tensor for async extraction.
   */
  private calculateContrastTensor(faceRegion: tf.Tensor2D): tf.Tensor {
    const mean = faceRegion.mean();
    const squaredDiff = tf.square(tf.sub(faceRegion, mean));
    const variance = squaredDiff.mean();

    // Clean up intermediate tensors
    mean.dispose();
    squaredDiff.dispose();

    return variance;
  }

  /**
   * Get default result when analysis fails.
   */
  private getDefaultResult(): ImageQualityResult {
    return {
      sharpness: 0.5,
      brightness: 0.5,
      contrast: 0.5,
      faceRegionExtracted: false,
    };
  }
}

/**
 * Singleton instance for convenience.
 */
export const imageQualityAnalyzer = new ImageQualityAnalyzer();

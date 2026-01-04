import * as tf from '../../dist/tfjs.esm';

import { FaceDetection } from '../classes/FaceDetection';
import { TNetInput, toNetInput } from '../dom/index';

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

    return tf.tidy(() => {
      // Convert to grayscale for quality analysis
      const grayscale = this.toGrayscale(inputTensor as tf.Tensor3D);

      // Extract face region
      const faceRegion = this.extractFaceRegion(grayscale, detection, netInput.getInputWidth(0), netInput.getInputHeight(0));
      if (!faceRegion) {
        return this.getDefaultResult();
      }

      // Calculate metrics
      const sharpness = this.calculateSharpness(faceRegion);
      const brightness = this.calculateBrightness(faceRegion);
      const contrast = this.calculateContrast(faceRegion);

      return {
        sharpness,
        brightness,
        contrast,
        faceRegionExtracted: true,
      };
    });
  }

  /**
   * Convert RGB tensor to grayscale.
   */
  private toGrayscale(tensor: tf.Tensor3D): tf.Tensor2D {
    return tf.tidy(() => {
      // Normalize to 0-1 if needed
      let normalized = tensor;
      const maxVal = tensor.max().dataSync()[0];
      if (maxVal && maxVal > 1) {
        normalized = tf.div(tensor, 255) as tf.Tensor3D;
      }

      // RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
      const [r, g, b] = tf.split(normalized, 3, 2);
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
    return tf.tidy(() => {
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
        const sliced = tf.slice(grayscale, [y1, x1], [height, width]);
        return sliced as tf.Tensor2D;
      } catch {
        return null;
      }
    });
  }

  /**
   * Calculate sharpness using Laplacian variance.
   * Higher variance = sharper image.
   */
  private calculateSharpness(faceRegion: tf.Tensor2D): number {
    return tf.tidy(() => {
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
      const variance = squaredDiff.mean().dataSync()[0] ?? 0;

      // Normalize to 0-1 range
      // Typical variance range for sharp images: 0.001-0.01
      // For blurry images: < 0.001
      const normalizedSharpness = Math.min(1, Math.max(0, variance * 100));

      return normalizedSharpness;
    });
  }

  /**
   * Calculate brightness as mean pixel intensity.
   */
  private calculateBrightness(faceRegion: tf.Tensor2D): number {
    return tf.tidy(() => {
      const meanBrightness = faceRegion.mean().dataSync()[0] ?? 0.5;
      // Values are already 0-1 from grayscale conversion
      return Math.min(1, Math.max(0, meanBrightness));
    });
  }

  /**
   * Calculate contrast as standard deviation of pixel intensities.
   */
  private calculateContrast(faceRegion: tf.Tensor2D): number {
    return tf.tidy(() => {
      const mean = faceRegion.mean();
      const squaredDiff = tf.square(tf.sub(faceRegion, mean));
      const variance = squaredDiff.mean().dataSync()[0] ?? 0;
      const stdDev = Math.sqrt(variance);

      // Normalize: good contrast has stdDev around 0.2-0.3
      // Low contrast: < 0.1, High contrast: > 0.4
      const normalizedContrast = Math.min(1, stdDev * 4);

      return normalizedContrast;
    });
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

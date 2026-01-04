/**
 * Image Augmentation for training data.
 *
 * Provides various augmentation techniques to increase
 * training data diversity and improve model robustness.
 *
 * @example
 * ```typescript
 * import { ImageAugmenter } from '@vladmandic/face-api/training';
 *
 * const augmenter = new ImageAugmenter({
 *   flipHorizontal: true,
 *   rotation: { min: -15, max: 15 },
 *   brightness: { min: 0.8, max: 1.2 },
 *   contrast: { min: 0.8, max: 1.2 }
 * });
 *
 * // Augment a single image
 * const augmented = await augmenter.augment(imageTensor);
 *
 * // Create augmented dataset
 * const augmentedDataset = await augmenter.augmentDataset(images, 5);
 * ```
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Range for numeric augmentation parameters.
 */
export interface AugmentRange {
  min: number;
  max: number;
}

/**
 * Image augmentation configuration.
 */
export interface AugmentationConfig {
  /** Random horizontal flip (default: true) */
  flipHorizontal?: boolean;
  /** Random vertical flip (default: false) */
  flipVertical?: boolean;
  /** Rotation range in degrees */
  rotation?: AugmentRange;
  /** Brightness adjustment range (1.0 = no change) */
  brightness?: AugmentRange;
  /** Contrast adjustment range (1.0 = no change) */
  contrast?: AugmentRange;
  /** Saturation adjustment range (1.0 = no change) */
  saturation?: AugmentRange;
  /** Hue shift range (0-1, wraps around) */
  hue?: AugmentRange;
  /** Random crop percentage (0-1) */
  cropScale?: AugmentRange;
  /** Gaussian noise standard deviation */
  noise?: number;
  /** Random zoom range */
  zoom?: AugmentRange;
  /** Probability of applying each augmentation (0-1, default: 0.5) */
  probability?: number;
}

/**
 * Augmentation result with applied transformations.
 */
export interface AugmentationResult {
  /** Augmented image tensor */
  image: tf.Tensor3D;
  /** Applied transformations */
  transformations: string[];
}

/**
 * Image Augmenter for training data enhancement.
 */
export class ImageAugmenter {
  private _config: Required<AugmentationConfig>;

  constructor(config: AugmentationConfig = {}) {
    this._config = {
      flipHorizontal: config.flipHorizontal ?? true,
      flipVertical: config.flipVertical ?? false,
      rotation: config.rotation ?? { min: -10, max: 10 },
      brightness: config.brightness ?? { min: 0.9, max: 1.1 },
      contrast: config.contrast ?? { min: 0.9, max: 1.1 },
      saturation: config.saturation ?? { min: 0.9, max: 1.1 },
      hue: config.hue ?? { min: -0.05, max: 0.05 },
      cropScale: config.cropScale ?? { min: 0.9, max: 1.0 },
      noise: config.noise ?? 0,
      zoom: config.zoom ?? { min: 0.95, max: 1.05 },
      probability: config.probability ?? 0.5
    };
  }

  /**
   * Apply random augmentations to an image.
   */
  public augment(image: tf.Tensor3D): AugmentationResult {
    const transformations: string[] = [];
    let result = image.clone();

    // Horizontal flip
    if (this._config.flipHorizontal && this._shouldApply()) {
      const flipped = tf.image.flipLeftRight(result);
      result.dispose();
      result = flipped as tf.Tensor3D;
      transformations.push('flipHorizontal');
    }

    // Vertical flip
    if (this._config.flipVertical && this._shouldApply()) {
      const flipped = this._flipVertical(result);
      result.dispose();
      result = flipped;
      transformations.push('flipVertical');
    }

    // Rotation
    if (this._config.rotation && this._shouldApply()) {
      const angle = this._randomInRange(this._config.rotation);
      const rotated = this._rotate(result, angle);
      result.dispose();
      result = rotated;
      transformations.push(`rotation:${angle.toFixed(1)}`);
    }

    // Brightness
    if (this._config.brightness && this._shouldApply()) {
      const factor = this._randomInRange(this._config.brightness);
      const adjusted = this._adjustBrightness(result, factor);
      result.dispose();
      result = adjusted;
      transformations.push(`brightness:${factor.toFixed(2)}`);
    }

    // Contrast
    if (this._config.contrast && this._shouldApply()) {
      const factor = this._randomInRange(this._config.contrast);
      const adjusted = this._adjustContrast(result, factor);
      result.dispose();
      result = adjusted;
      transformations.push(`contrast:${factor.toFixed(2)}`);
    }

    // Saturation
    if (this._config.saturation && this._shouldApply()) {
      const factor = this._randomInRange(this._config.saturation);
      const adjusted = this._adjustSaturation(result, factor);
      result.dispose();
      result = adjusted;
      transformations.push(`saturation:${factor.toFixed(2)}`);
    }

    // Hue
    if (this._config.hue && this._shouldApply()) {
      const shift = this._randomInRange(this._config.hue);
      const adjusted = this._adjustHue(result, shift);
      result.dispose();
      result = adjusted;
      transformations.push(`hue:${shift.toFixed(2)}`);
    }

    // Random crop
    if (this._config.cropScale && this._shouldApply()) {
      const scale = this._randomInRange(this._config.cropScale);
      if (scale < 1.0) {
        const cropped = this._randomCrop(result, scale);
        result.dispose();
        result = cropped;
        transformations.push(`crop:${scale.toFixed(2)}`);
      }
    }

    // Gaussian noise
    if (this._config.noise > 0 && this._shouldApply()) {
      const noisy = this._addNoise(result, this._config.noise);
      result.dispose();
      result = noisy;
      transformations.push(`noise:${this._config.noise.toFixed(3)}`);
    }

    // Zoom
    if (this._config.zoom && this._shouldApply()) {
      const factor = this._randomInRange(this._config.zoom);
      const zoomed = this._zoom(result, factor);
      result.dispose();
      result = zoomed;
      transformations.push(`zoom:${factor.toFixed(2)}`);
    }

    return { image: result, transformations };
  }

  /**
   * Generate multiple augmented versions of an image.
   */
  public augmentMultiple(image: tf.Tensor3D, count: number): AugmentationResult[] {
    const results: AugmentationResult[] = [];

    for (let i = 0; i < count; i++) {
      results.push(this.augment(image));
    }

    return results;
  }

  /**
   * Augment a batch of images.
   * Note: Caller is responsible for disposing the returned tensor.
   */
  public augmentBatch(
    images: tf.Tensor4D,
    augmentationsPerImage = 1
  ): tf.Tensor4D {
    const numImages = images.shape[0];
    const augmented: tf.Tensor3D[] = [];

    for (let i = 0; i < numImages; i++) {
      const image = tf.slice(images, [i, 0, 0, 0], [1, -1, -1, -1]);
      const squeezed = tf.squeeze(image, [0]) as tf.Tensor3D;
      image.dispose();

      // Add original
      augmented.push(squeezed.clone());

      // Add augmented versions
      for (let j = 0; j < augmentationsPerImage; j++) {
        const result = this.augment(squeezed);
        augmented.push(result.image);
      }

      squeezed.dispose();
    }

    const stacked = tf.stack(augmented) as tf.Tensor4D;

    // Cleanup intermediate tensors
    for (const t of augmented) {
      t.dispose();
    }

    return stacked;
  }

  /**
   * Check if augmentation should be applied based on probability.
   */
  private _shouldApply(): boolean {
    return Math.random() < this._config.probability;
  }

  /**
   * Get random value within range.
   */
  private _randomInRange(range: AugmentRange): number {
    return range.min + Math.random() * (range.max - range.min);
  }

  /**
   * Flip image vertically.
   */
  private _flipVertical(image: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => tf.reverse(image, 0) as tf.Tensor3D);
  }

  /**
   * Rotate image by angle in degrees.
   */
  private _rotate(image: tf.Tensor3D, angleDegrees: number): tf.Tensor3D {
    return tf.tidy(() => {
      const [height, width] = image.shape;
      const angleRadians = (angleDegrees * Math.PI) / 180;

      // Create rotation matrix
      const cos = Math.cos(angleRadians);
      const sin = Math.sin(angleRadians);

      // Center of image
      const cx = width / 2;
      const cy = height / 2;

      // Use image transform
      const transforms = [
        cos, -sin, cx * (1 - cos) + cy * sin,
        sin, cos, cy * (1 - cos) - cx * sin,
        0, 0
      ];

      const expanded = tf.expandDims(image, 0) as tf.Tensor4D;
      const rotated = tf.image.transform(
        expanded,
        [transforms],
        'bilinear',
        'constant',
        0
      );

      return tf.squeeze(rotated, [0]) as tf.Tensor3D;
    });
  }

  /**
   * Adjust image brightness.
   */
  private _adjustBrightness(image: tf.Tensor3D, factor: number): tf.Tensor3D {
    return tf.tidy(() => {
      const adjusted = tf.mul(image, factor);
      return tf.clipByValue(adjusted, 0, 255) as tf.Tensor3D;
    });
  }

  /**
   * Adjust image contrast.
   */
  private _adjustContrast(image: tf.Tensor3D, factor: number): tf.Tensor3D {
    return tf.tidy(() => {
      const mean = tf.mean(image);
      const adjusted = tf.add(tf.mul(tf.sub(image, mean), factor), mean);
      return tf.clipByValue(adjusted, 0, 255) as tf.Tensor3D;
    });
  }

  /**
   * Adjust image saturation.
   */
  private _adjustSaturation(image: tf.Tensor3D, factor: number): tf.Tensor3D {
    return tf.tidy(() => {
      // Convert to grayscale
      const gray = tf.mean(image, -1, true);
      const grayRgb = tf.tile(gray, [1, 1, 3]);

      // Interpolate between grayscale and color
      const adjusted = tf.add(
        tf.mul(grayRgb, 1 - factor),
        tf.mul(image, factor)
      );

      return tf.clipByValue(adjusted, 0, 255) as tf.Tensor3D;
    });
  }

  /**
   * Adjust image hue.
   */
  private _adjustHue(image: tf.Tensor3D, shift: number): tf.Tensor3D {
    return tf.tidy(() => {
      // Simple hue approximation - shift RGB channels
      const [r, g, b] = tf.split(image, 3, -1);

      // Rotate through channels based on shift
      const shifted = shift > 0
        ? tf.concat([g, b, r], -1)
        : tf.concat([b, r, g], -1);

      // Blend with original based on shift magnitude
      const blendFactor = Math.abs(shift);
      const blended = tf.add(
        tf.mul(image, 1 - blendFactor),
        tf.mul(shifted, blendFactor)
      );

      return tf.clipByValue(blended, 0, 255) as tf.Tensor3D;
    });
  }

  /**
   * Random crop with resize back to original size.
   */
  private _randomCrop(image: tf.Tensor3D, scale: number): tf.Tensor3D {
    return tf.tidy(() => {
      const [height, width] = image.shape;
      const cropHeight = Math.max(1, Math.floor(height * scale));
      const cropWidth = Math.max(1, Math.floor(width * scale));

      const maxY = Math.max(0, height - cropHeight);
      const maxX = Math.max(0, width - cropWidth);

      const y = maxY > 0 ? Math.floor(Math.random() * maxY) : 0;
      const x = maxX > 0 ? Math.floor(Math.random() * maxX) : 0;

      // Crop
      const cropped = tf.slice(image, [y, x, 0], [cropHeight, cropWidth, -1]);

      // Resize back to original size
      const resized = tf.image.resizeBilinear(
        cropped as tf.Tensor3D,
        [height, width]
      );

      return resized as tf.Tensor3D;
    });
  }

  /**
   * Add Gaussian noise to image.
   */
  private _addNoise(image: tf.Tensor3D, stdDev: number): tf.Tensor3D {
    return tf.tidy(() => {
      const noise = tf.randomNormal(image.shape, 0, stdDev * 255);
      const noisy = tf.add(image, noise);
      return tf.clipByValue(noisy, 0, 255) as tf.Tensor3D;
    });
  }

  /**
   * Zoom image (crop and resize).
   */
  private _zoom(image: tf.Tensor3D, factor: number): tf.Tensor3D {
    return tf.tidy(() => {
      const [height, width] = image.shape;

      if (factor === 1.0) return image.clone();

      if (factor > 1.0) {
        // Zoom in - crop center
        const cropHeight = Math.floor(height / factor);
        const cropWidth = Math.floor(width / factor);
        const y = Math.floor((height - cropHeight) / 2);
        const x = Math.floor((width - cropWidth) / 2);

        const cropped = tf.slice(image, [y, x, 0], [cropHeight, cropWidth, -1]);
        return tf.image.resizeBilinear(cropped as tf.Tensor3D, [height, width]) as tf.Tensor3D;
      } else {
        // Zoom out - pad and resize
        const paddedHeight = Math.floor(height / factor);
        const paddedWidth = Math.floor(width / factor);
        const padY = Math.floor((paddedHeight - height) / 2);
        const padX = Math.floor((paddedWidth - width) / 2);

        const padded = tf.pad(image, [
          [padY, paddedHeight - height - padY],
          [padX, paddedWidth - width - padX],
          [0, 0]
        ]);

        return tf.image.resizeBilinear(padded as tf.Tensor3D, [height, width]) as tf.Tensor3D;
      }
    });
  }

  /**
   * Get current configuration.
   */
  public getConfig(): AugmentationConfig {
    return { ...this._config };
  }

  /**
   * Update configuration.
   */
  public setConfig(config: Partial<AugmentationConfig>): void {
    Object.assign(this._config, config);
  }
}

/**
 * Create a pipeline of augmentations.
 */
export class AugmentationPipeline {
  private _stages: Array<{
    name: string;
    augmenter: ImageAugmenter;
    probability: number;
  }> = [];

  /**
   * Add an augmentation stage.
   */
  public addStage(
    name: string,
    config: AugmentationConfig,
    probability = 1.0
  ): AugmentationPipeline {
    this._stages.push({
      name,
      augmenter: new ImageAugmenter({ ...config, probability: 1.0 }),
      probability
    });
    return this;
  }

  /**
   * Apply all pipeline stages.
   */
  public apply(image: tf.Tensor3D): AugmentationResult {
    let current = image.clone();
    const allTransformations: string[] = [];

    for (const stage of this._stages) {
      if (Math.random() < stage.probability) {
        const result = stage.augmenter.augment(current);
        current.dispose();
        current = result.image;
        allTransformations.push(
          ...result.transformations.map(t => `${stage.name}:${t}`)
        );
      }
    }

    return { image: current, transformations: allTransformations };
  }

  /**
   * Get pipeline stages.
   */
  public getStages(): string[] {
    return this._stages.map(s => s.name);
  }
}

/**
 * Preset augmentation configurations.
 */
export const AugmentationPresets = {
  /** Light augmentation - minimal changes */
  light: {
    flipHorizontal: true,
    flipVertical: false,
    rotation: { min: -5, max: 5 },
    brightness: { min: 0.95, max: 1.05 },
    contrast: { min: 0.95, max: 1.05 },
    probability: 0.3
  } as AugmentationConfig,

  /** Medium augmentation - moderate changes */
  medium: {
    flipHorizontal: true,
    flipVertical: false,
    rotation: { min: -15, max: 15 },
    brightness: { min: 0.8, max: 1.2 },
    contrast: { min: 0.8, max: 1.2 },
    saturation: { min: 0.8, max: 1.2 },
    cropScale: { min: 0.85, max: 1.0 },
    probability: 0.5
  } as AugmentationConfig,

  /** Heavy augmentation - significant changes */
  heavy: {
    flipHorizontal: true,
    flipVertical: true,
    rotation: { min: -30, max: 30 },
    brightness: { min: 0.6, max: 1.4 },
    contrast: { min: 0.6, max: 1.4 },
    saturation: { min: 0.5, max: 1.5 },
    hue: { min: -0.1, max: 0.1 },
    cropScale: { min: 0.7, max: 1.0 },
    noise: 0.02,
    zoom: { min: 0.8, max: 1.2 },
    probability: 0.7
  } as AugmentationConfig,

  /** Face-specific augmentation */
  face: {
    flipHorizontal: true,
    flipVertical: false,
    rotation: { min: -10, max: 10 },
    brightness: { min: 0.8, max: 1.2 },
    contrast: { min: 0.85, max: 1.15 },
    saturation: { min: 0.9, max: 1.1 },
    cropScale: { min: 0.9, max: 1.0 },
    noise: 0.01,
    probability: 0.5
  } as AugmentationConfig
};

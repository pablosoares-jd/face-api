import * as tf from '@tensorflow/tfjs';

/**
 * Expected input range for different normalization schemes.
 */
export interface InputRangeSpec {
  /** Minimum expected value */
  min: number;
  /** Maximum expected value */
  max: number;
  /** Description of the expected range */
  description: string;
}

/**
 * Common input ranges used by different models.
 */
export const INPUT_RANGES = {
  /** Standard image range [0, 255] */
  IMAGE_UINT8: { min: 0, max: 255, description: '[0, 255] (uint8 image)' } as InputRangeSpec,
  /** Normalized range [0, 1] */
  NORMALIZED_01: { min: 0, max: 1, description: '[0, 1] (normalized)' } as InputRangeSpec,
  /** Centered range [-1, 1] */
  CENTERED: { min: -1, max: 1, description: '[-1, 1] (centered)' } as InputRangeSpec,
} as const;

/**
 * Validation result for input tensors.
 */
export interface InputValidationResult {
  /** Whether the input is valid */
  isValid: boolean;
  /** Actual minimum value found */
  actualMin: number;
  /** Actual maximum value found */
  actualMax: number;
  /** Warning message if input is outside expected range */
  warning?: string;
  /** Suggested fix */
  suggestion?: string;
}

/**
 * Options for input validation.
 */
export interface ValidationOptions {
  /** Whether to log warnings to console. Default: true */
  logWarnings?: boolean;
  /** Model name for context in warning messages */
  modelName?: string;
  /** Tolerance for range checking (percentage). Default: 0.1 (10%) */
  tolerance?: number;
}

/**
 * Validate that a tensor's values are within the expected range.
 * Uses async data() to avoid GPU blocking.
 *
 * @param tensor The input tensor to validate
 * @param expectedRange The expected input range
 * @param options Validation options
 * @returns Validation result
 *
 * @example
 * ```typescript
 * const result = await validateInputRange(inputTensor, INPUT_RANGES.IMAGE_UINT8, {
 *   modelName: 'SsdMobilenetv1'
 * });
 * if (!result.isValid) {
 *   console.warn(result.warning);
 * }
 * ```
 */
export async function validateInputRange(
  tensor: tf.Tensor,
  expectedRange: InputRangeSpec,
  options: ValidationOptions = {},
): Promise<InputValidationResult> {
  const {
    logWarnings = true,
    modelName = 'Model',
    tolerance = 0.1,
  } = options;

  // Calculate min and max asynchronously
  const minTensor = tensor.min();
  const maxTensor = tensor.max();

  try {
    const [minData, maxData] = await Promise.all([
      minTensor.data(),
      maxTensor.data(),
    ]);

    const actualMin = minData[0] ?? 0;
    const actualMax = maxData[0] ?? 0;

    // Calculate tolerance bounds
    const rangeSize = expectedRange.max - expectedRange.min;
    const toleranceValue = rangeSize * tolerance;
    const minBound = expectedRange.min - toleranceValue;
    const maxBound = expectedRange.max + toleranceValue;

    const isValid = actualMin >= minBound && actualMax <= maxBound;

    const result: InputValidationResult = {
      isValid,
      actualMin,
      actualMax,
    };

    if (!isValid) {
      result.warning = `${modelName}: Input values [${actualMin.toFixed(2)}, ${actualMax.toFixed(2)}] `
        + `are outside expected range ${expectedRange.description}`;

      // Generate suggestion based on actual range
      if (actualMax > 1 && actualMax <= 255 && expectedRange.max === 1) {
        result.suggestion = 'Input appears to be in [0, 255] range. Consider dividing by 255.';
      } else if (actualMin >= 0 && actualMax <= 1 && expectedRange.max === 255) {
        result.suggestion = 'Input appears to be normalized [0, 1]. Consider multiplying by 255.';
      } else if (actualMin < -1 || actualMax > 1) {
        result.suggestion = 'Input may need normalization. Check preprocessing pipeline.';
      }

      if (logWarnings) {
        console.warn(result.warning);
        if (result.suggestion) {
          console.warn(`  Suggestion: ${result.suggestion}`);
        }
      }
    }

    return result;
  } finally {
    minTensor.dispose();
    maxTensor.dispose();
  }
}

/**
 * Validate input synchronously using sampled values (faster but less accurate).
 * Use this for quick sanity checks where full tensor scan is not needed.
 *
 * @param tensor The input tensor
 * @param expectedRange The expected input range
 * @param sampleSize Number of values to sample. Default: 100
 * @returns Partial validation result
 */
export function validateInputRangeQuick(
  tensor: tf.Tensor,
  expectedRange: InputRangeSpec,
  sampleSize = 100,
): { likelyValid: boolean; sampledMin: number; sampledMax: number } {
  // This uses slice to sample first N values - faster than full tensor scan
  const flatTensor = tensor.flatten();
  const actualSize = flatTensor.shape[0];
  const samplesToTake = Math.min(sampleSize, actualSize);

  const sampled = tf.tidy(() => tf.slice(flatTensor, 0, samplesToTake));

  // Use dataSync here since it's a small sample
  const data = sampled.dataSync();
  flatTensor.dispose();
  sampled.dispose();

  let sampledMin = Infinity;
  let sampledMax = -Infinity;

  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    if (val !== undefined) {
      if (val < sampledMin) sampledMin = val;
      if (val > sampledMax) sampledMax = val;
    }
  }

  const tolerance = (expectedRange.max - expectedRange.min) * 0.1;
  const likelyValid = sampledMin >= expectedRange.min - tolerance
    && sampledMax <= expectedRange.max + tolerance;

  return { likelyValid, sampledMin, sampledMax };
}

/**
 * Check if tensor shape is valid for face detection models.
 *
 * @param tensor The input tensor
 * @param expectedChannels Expected number of channels (3 for RGB, 4 for RGBA)
 * @returns Whether shape is valid with details
 */
export function validateTensorShape(
  tensor: tf.Tensor,
  expectedChannels: 3 | 4 = 3,
): { isValid: boolean; shape: number[]; warning?: string } {
  const shape = tensor.shape;
  const rank = shape.length;

  // Valid shapes: [height, width, channels] or [batch, height, width, channels]
  if (rank !== 3 && rank !== 4) {
    return {
      isValid: false,
      shape,
      warning: `Invalid tensor rank ${rank}. Expected 3 (HWC) or 4 (NHWC).`,
    };
  }

  const channelIdx = rank - 1;
  const channels = shape[channelIdx];

  if (channels !== expectedChannels && channels !== 3 && channels !== 4) {
    return {
      isValid: false,
      shape,
      warning: `Invalid number of channels: ${channels}. Expected ${expectedChannels}.`,
    };
  }

  const heightIdx = rank === 3 ? 0 : 1;
  const widthIdx = rank === 3 ? 1 : 2;
  const height = shape[heightIdx];
  const width = shape[widthIdx];

  if (!height || !width || height <= 0 || width <= 0) {
    return {
      isValid: false,
      shape,
      warning: `Invalid dimensions: ${height}x${width}.`,
    };
  }

  return { isValid: true, shape };
}

/**
 * Normalization helper functions.
 */
export const normalize = {
  /**
   * Normalize [0, 255] to [0, 1].
   */
  toUnit: (tensor: tf.Tensor): tf.Tensor => tf.div(tensor, 255),

  /**
   * Normalize [0, 255] to [-1, 1].
   */
  toCentered: (tensor: tf.Tensor): tf.Tensor => tf.sub(tf.div(tensor, 127.5), 1),

  /**
   * Normalize with mean RGB subtraction (used by FaceRecognitionNet).
   * @param tensor Input tensor in [0, 255] range
   * @param meanRGB Mean RGB values to subtract [R, G, B]
   */
  withMeanSubtraction: (tensor: tf.Tensor, meanRGB: [number, number, number]): tf.Tensor => tf.tidy(() => {
    const meanTensor = tf.tensor1d(meanRGB);
    return tf.sub(tensor, meanTensor);
  }),

  /**
   * AdaFace normalization: (x - 127.5) / 128.
   */
  adaFace: (tensor: tf.Tensor): tf.Tensor => tf.div(tf.sub(tensor, 127.5), 128),
};

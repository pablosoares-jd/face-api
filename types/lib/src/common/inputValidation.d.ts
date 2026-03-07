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
export declare const INPUT_RANGES: {
    /** Standard image range [0, 255] */
    readonly IMAGE_UINT8: InputRangeSpec;
    /** Normalized range [0, 1] */
    readonly NORMALIZED_01: InputRangeSpec;
    /** Centered range [-1, 1] */
    readonly CENTERED: InputRangeSpec;
};
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
export declare function validateInputRange(tensor: tf.Tensor, expectedRange: InputRangeSpec, options?: ValidationOptions): Promise<InputValidationResult>;
/**
 * Validate input synchronously using sampled values (faster but less accurate).
 * Use this for quick sanity checks where full tensor scan is not needed.
 *
 * @param tensor The input tensor
 * @param expectedRange The expected input range
 * @param sampleSize Number of values to sample. Default: 100
 * @returns Partial validation result
 */
export declare function validateInputRangeQuick(tensor: tf.Tensor, expectedRange: InputRangeSpec, sampleSize?: number): {
    likelyValid: boolean;
    sampledMin: number;
    sampledMax: number;
};
/**
 * Check if tensor shape is valid for face detection models.
 *
 * @param tensor The input tensor
 * @param expectedChannels Expected number of channels (3 for RGB, 4 for RGBA)
 * @returns Whether shape is valid with details
 */
export declare function validateTensorShape(tensor: tf.Tensor, expectedChannels?: 3 | 4): {
    isValid: boolean;
    shape: number[];
    warning?: string;
};
/**
 * Normalization helper functions.
 */
export declare const normalize: {
    /**
     * Normalize [0, 255] to [0, 1].
     */
    toUnit: (tensor: tf.Tensor) => tf.Tensor;
    /**
     * Normalize [0, 255] to [-1, 1].
     */
    toCentered: (tensor: tf.Tensor) => tf.Tensor;
    /**
     * Normalize with mean RGB subtraction (used by FaceRecognitionNet).
     * @param tensor Input tensor in [0, 255] range
     * @param meanRGB Mean RGB values to subtract [R, G, B]
     */
    withMeanSubtraction: (tensor: tf.Tensor, meanRGB: [number, number, number]) => tf.Tensor;
    /**
     * AdaFace normalization: (x - 127.5) / 128.
     */
    adaFace: (tensor: tf.Tensor) => tf.Tensor;
};

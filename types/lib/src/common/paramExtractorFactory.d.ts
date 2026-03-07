import * as tf from '@tensorflow/tfjs';
import type { ParamMapping, ConvParams, FCParams } from './types';
/**
 * Batch normalization parameters.
 */
export interface BatchNormParams {
    mean: tf.Tensor1D;
    variance: tf.Tensor1D;
    scale: tf.Tensor1D;
    offset: tf.Tensor1D;
}
/**
 * Depthwise convolution parameters with batch norm.
 */
export interface DepthwiseConvParams {
    filters: tf.Tensor4D;
    batchNorm: BatchNormParams;
}
/**
 * Separable convolution parameters (depthwise + pointwise).
 */
export interface SeparableConvBlockParams {
    depthwise: DepthwiseConvParams;
    pointwise: ConvParams;
}
/**
 * Residual block parameters.
 */
export interface ResidualBlockParams {
    conv1: ConvParams;
    conv2: ConvParams;
    shortcut?: ConvParams;
}
/**
 * Configuration for creating a parameter extractor.
 */
export interface ExtractorConfig {
    /** Initial weights as Float32Array */
    weights: Float32Array;
    /** Whether to track parameter mappings (default: true) */
    trackMappings?: boolean;
    /** Whether to throw on remaining weights (default: true) */
    strictMode?: boolean;
}
/**
 * Result from parameter extraction.
 */
export interface ExtractionResult<T> {
    /** Extracted parameters */
    params: T;
    /** Parameter mappings for weight disposal */
    paramMappings: ParamMapping[];
    /** Remaining weights after extraction */
    remainingWeights: Float32Array;
}
/**
 * Unified parameter extractor factory.
 * Provides common extraction functions for neural network weights.
 */
export declare function createParamExtractor(config: ExtractorConfig): {
    extractWeights: (numWeights: number) => Float32Array;
    getRemainingWeights: () => Float32Array;
    addMapping: (paramPath: string, originalPath?: string) => void;
    tensor1d: (size: number, prefix: string, name?: string) => tf.Tensor1D;
    tensor2d: (rows: number, cols: number, prefix: string, name?: string) => tf.Tensor2D;
    tensor4d: (height: number, width: number, inChannels: number, outChannels: number, prefix: string, name?: string) => tf.Tensor4D;
    convParams: (filterSize: number, inChannels: number, outChannels: number, prefix: string) => ConvParams;
    fcParams: (inputSize: number, outputSize: number, prefix: string) => FCParams;
    batchNormParams: (channels: number, prefix: string) => BatchNormParams;
    depthwiseConvParams: (channels: number, filterSize: number, prefix: string) => DepthwiseConvParams;
    separableConvParams: (inChannels: number, outChannels: number, filterSize: number, prefix: string) => SeparableConvBlockParams;
    residualBlockParams: (inChannels: number, outChannels: number, filterSize: number, prefix: string, hasShortcut?: boolean) => ResidualBlockParams;
    verifyComplete: () => void;
    getResult: <T>(params: T) => ExtractionResult<T>;
    readonly paramMappings: ParamMapping[];
};
/**
 * Type for the parameter extractor.
 */
export type ParamExtractor = ReturnType<typeof createParamExtractor>;
/**
 * Create a simple extractor for flat weight arrays.
 * Useful for simple models without complex nesting.
 */
export declare function createSimpleExtractor(weights: Float32Array): {
    extractWeights: (numWeights: number) => Float32Array;
    getRemainingWeights: () => Float32Array;
    addMapping: (paramPath: string, originalPath?: string) => void;
    tensor1d: (size: number, prefix: string, name?: string) => tf.Tensor1D;
    tensor2d: (rows: number, cols: number, prefix: string, name?: string) => tf.Tensor2D;
    tensor4d: (height: number, width: number, inChannels: number, outChannels: number, prefix: string, name?: string) => tf.Tensor4D;
    convParams: (filterSize: number, inChannels: number, outChannels: number, prefix: string) => ConvParams;
    fcParams: (inputSize: number, outputSize: number, prefix: string) => FCParams;
    batchNormParams: (channels: number, prefix: string) => BatchNormParams;
    depthwiseConvParams: (channels: number, filterSize: number, prefix: string) => DepthwiseConvParams;
    separableConvParams: (inChannels: number, outChannels: number, filterSize: number, prefix: string) => SeparableConvBlockParams;
    residualBlockParams: (inChannels: number, outChannels: number, filterSize: number, prefix: string, hasShortcut?: boolean) => ResidualBlockParams;
    verifyComplete: () => void;
    getResult: <T>(params: T) => ExtractionResult<T>;
    readonly paramMappings: ParamMapping[];
};
/**
 * Utility to count total parameters in a model.
 */
export declare function countParameters(params: Record<string, unknown>): number;
/**
 * Utility to calculate model size in MB.
 */
export declare function calculateModelSizeMB(params: Record<string, unknown>): number;

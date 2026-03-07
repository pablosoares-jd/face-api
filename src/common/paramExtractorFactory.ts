import * as tf from '@tensorflow/tfjs';

import type { ParamMapping, ConvParams, FCParams } from './types';
import { extractWeightsFactory } from './extractWeightsFactory';

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
export function createParamExtractor(config: ExtractorConfig) {
  const { weights, trackMappings = true, strictMode = true } = config;
  const { extractWeights, getRemainingWeights } = extractWeightsFactory(weights);
  const paramMappings: ParamMapping[] = [];

  /**
   * Add a parameter mapping if tracking is enabled.
   */
  function addMapping(paramPath: string, originalPath?: string): void {
    if (trackMappings) {
      paramMappings.push({ paramPath, originalPath });
    }
  }

  /**
   * Extract a 1D tensor (e.g., bias, scale).
   */
  function tensor1d(size: number, prefix: string, name = 'tensor'): tf.Tensor1D {
    const data = extractWeights(size);
    addMapping(`${prefix}/${name}`);
    return tf.tensor1d(data);
  }

  /**
   * Extract a 2D tensor (e.g., FC weights).
   */
  function tensor2d(rows: number, cols: number, prefix: string, name = 'weights'): tf.Tensor2D {
    const data = extractWeights(rows * cols);
    addMapping(`${prefix}/${name}`);
    return tf.tensor2d(data, [rows, cols]);
  }

  /**
   * Extract a 4D tensor (e.g., conv filters).
   */
  function tensor4d(
    height: number,
    width: number,
    inChannels: number,
    outChannels: number,
    prefix: string,
    name = 'filters',
  ): tf.Tensor4D {
    const data = extractWeights(height * width * inChannels * outChannels);
    addMapping(`${prefix}/${name}`);
    return tf.tensor4d(data, [height, width, inChannels, outChannels]);
  }

  /**
   * Extract convolutional layer parameters (filters + bias).
   */
  function convParams(
    filterSize: number,
    inChannels: number,
    outChannels: number,
    prefix: string,
  ): ConvParams {
    const filters = tensor4d(filterSize, filterSize, inChannels, outChannels, prefix, 'filters');
    const bias = tensor1d(outChannels, prefix, 'bias');
    return { filters, bias };
  }

  /**
   * Extract fully connected layer parameters.
   */
  function fcParams(inputSize: number, outputSize: number, prefix: string): FCParams {
    const fcWeights = tensor2d(inputSize, outputSize, prefix, 'weights');
    const bias = tensor1d(outputSize, prefix, 'bias');
    return { weights: fcWeights, bias };
  }

  /**
   * Extract batch normalization parameters.
   */
  function batchNormParams(channels: number, prefix: string): BatchNormParams {
    const scale = tensor1d(channels, prefix, 'scale');
    const offset = tensor1d(channels, prefix, 'offset');
    const mean = tensor1d(channels, prefix, 'mean');
    const variance = tensor1d(channels, prefix, 'variance');
    return { scale, offset, mean, variance };
  }

  /**
   * Extract depthwise convolution parameters with batch norm.
   */
  function depthwiseConvParams(channels: number, filterSize: number, prefix: string): DepthwiseConvParams {
    // Depthwise conv has shape [filterSize, filterSize, channels, 1]
    const filtersData = extractWeights(filterSize * filterSize * channels);
    addMapping(`${prefix}/filters`);
    const filters = tf.tensor4d(filtersData, [filterSize, filterSize, channels, 1]);

    const batchNorm = batchNormParams(channels, `${prefix}/batchNorm`);

    return { filters, batchNorm };
  }

  /**
   * Extract separable convolution parameters.
   */
  function separableConvParams(
    inChannels: number,
    outChannels: number,
    filterSize: number,
    prefix: string,
  ): SeparableConvBlockParams {
    const depthwise = depthwiseConvParams(inChannels, filterSize, `${prefix}/depthwise`);
    const pointwise = convParams(1, inChannels, outChannels, `${prefix}/pointwise`);
    return { depthwise, pointwise };
  }

  /**
   * Extract residual block parameters.
   */
  function residualBlockParams(
    inChannels: number,
    outChannels: number,
    filterSize: number,
    prefix: string,
    hasShortcut = false,
  ): ResidualBlockParams {
    const conv1 = convParams(filterSize, inChannels, outChannels, `${prefix}/conv1`);
    const conv2 = convParams(filterSize, outChannels, outChannels, `${prefix}/conv2`);

    let shortcut: ConvParams | undefined;
    if (hasShortcut) {
      shortcut = convParams(1, inChannels, outChannels, `${prefix}/shortcut`);
    }

    return { conv1, conv2, shortcut };
  }

  /**
   * Verify all weights have been extracted.
   */
  function verifyComplete(): void {
    const remaining = getRemainingWeights();
    if (strictMode && remaining.length !== 0) {
      throw new Error(`Weights remaining after extraction: ${remaining.length}`);
    }
  }

  /**
   * Get extraction result.
   */
  function getResult<T>(params: T): ExtractionResult<T> {
    verifyComplete();
    return {
      params,
      paramMappings,
      remainingWeights: getRemainingWeights(),
    };
  }

  return {
    // Basic extractors
    extractWeights,
    getRemainingWeights,
    addMapping,

    // Tensor extractors
    tensor1d,
    tensor2d,
    tensor4d,

    // Layer extractors
    convParams,
    fcParams,
    batchNormParams,
    depthwiseConvParams,
    separableConvParams,
    residualBlockParams,

    // Finalization
    verifyComplete,
    getResult,

    // Access to mappings
    get paramMappings() {
      return paramMappings;
    },
  };
}

/**
 * Type for the parameter extractor.
 */
export type ParamExtractor = ReturnType<typeof createParamExtractor>;

/**
 * Create a simple extractor for flat weight arrays.
 * Useful for simple models without complex nesting.
 */
export function createSimpleExtractor(weights: Float32Array) {
  return createParamExtractor({ weights, trackMappings: false, strictMode: false });
}

/**
 * Utility to count total parameters in a model.
 */
export function countParameters(params: Record<string, unknown>): number {
  let count = 0;

  function traverse(obj: unknown): void {
    if (obj instanceof tf.Tensor) {
      count += obj.size;
    } else if (typeof obj === 'object' && obj !== null) {
      for (const value of Object.values(obj)) {
        traverse(value);
      }
    }
  }

  traverse(params);
  return count;
}

/**
 * Utility to calculate model size in MB.
 */
export function calculateModelSizeMB(params: Record<string, unknown>): number {
  const paramCount = countParameters(params);
  // Assuming float32 (4 bytes per parameter)
  return (paramCount * 4) / (1024 * 1024);
}

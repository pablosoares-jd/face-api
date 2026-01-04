import { ValidationError, InvalidDimensionsError } from '../errors';

/**
 * Configuration for image dimension validation
 */
export interface ImageDimensionConfig {
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
}

const DEFAULT_DIMENSION_CONFIG: ImageDimensionConfig = {
  minWidth: 1,
  minHeight: 1,
  maxWidth: 10000,
  maxHeight: 10000,
};

/**
 * Validates that a value is a valid number (not NaN, Infinity, or undefined)
 */
export function assertValidNumber(value: unknown, paramName: string): asserts value is number {
  if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) {
    throw new ValidationError(
      `Parameter "${paramName}" must be a valid finite number, received: ${typeof value === 'number' ? value : typeof value}`,
      { paramName, value }
    );
  }
}

/**
 * Validates that a value is a positive number
 */
export function assertPositiveNumber(value: unknown, paramName: string): asserts value is number {
  assertValidNumber(value, paramName);
  if (value <= 0) {
    throw new ValidationError(
      `Parameter "${paramName}" must be a positive number, received: ${value}`,
      { paramName, value }
    );
  }
}

/**
 * Validates that a value is a non-negative number
 */
export function assertNonNegativeNumber(value: unknown, paramName: string): asserts value is number {
  assertValidNumber(value, paramName);
  if (value < 0) {
    throw new ValidationError(
      `Parameter "${paramName}" must be a non-negative number, received: ${value}`,
      { paramName, value }
    );
  }
}

/**
 * Validates that a value is a valid probability (between 0 and 1)
 */
export function assertValidProbability(value: unknown, paramName: string): asserts value is number {
  assertValidNumber(value, paramName);
  if (value < 0 || value > 1) {
    throw new ValidationError(
      `Parameter "${paramName}" must be a probability between 0 and 1, received: ${value}`,
      { paramName, value }
    );
  }
}

/**
 * Validates image dimensions
 */
export function assertValidDimensions(
  width: number,
  height: number,
  config: ImageDimensionConfig = DEFAULT_DIMENSION_CONFIG
): void {
  const { minWidth = 1, minHeight = 1, maxWidth = 10000, maxHeight = 10000 } = config;

  if (width < minWidth || height < minHeight) {
    throw new InvalidDimensionsError(
      width,
      height,
      `Dimensions must be at least ${minWidth}x${minHeight}`
    );
  }

  if (width > maxWidth || height > maxHeight) {
    throw new InvalidDimensionsError(
      width,
      height,
      `Dimensions must not exceed ${maxWidth}x${maxHeight}`
    );
  }
}

/**
 * Validates that a value is a non-empty array
 */
export function assertNonEmptyArray<T>(value: unknown, paramName: string): asserts value is T[] {
  if (!Array.isArray(value)) {
    throw new ValidationError(
      `Parameter "${paramName}" must be an array, received: ${typeof value}`,
      { paramName, valueType: typeof value }
    );
  }
  if (value.length === 0) {
    throw new ValidationError(
      `Parameter "${paramName}" must be a non-empty array`,
      { paramName }
    );
  }
}

/**
 * Validates that two arrays have the same length
 */
export function assertArraysEqualLength(
  arr1: unknown[],
  arr2: unknown[],
  name1: string,
  name2: string
): void {
  if (arr1.length !== arr2.length) {
    throw new ValidationError(
      `Arrays "${name1}" and "${name2}" must have the same length. Got ${arr1.length} and ${arr2.length}`,
      { [name1 + 'Length']: arr1.length, [name2 + 'Length']: arr2.length }
    );
  }
}

/**
 * Validates that a value is defined (not null or undefined)
 */
export function assertDefined<T>(value: T | null | undefined, paramName: string): asserts value is T {
  if (value === null || value === undefined) {
    throw new ValidationError(
      `Parameter "${paramName}" is required but was ${value === null ? 'null' : 'undefined'}`,
      { paramName }
    );
  }
}

/**
 * Validates that a value is a string with content
 */
export function assertNonEmptyString(value: unknown, paramName: string): asserts value is string {
  if (typeof value !== 'string') {
    throw new ValidationError(
      `Parameter "${paramName}" must be a string, received: ${typeof value}`,
      { paramName, valueType: typeof value }
    );
  }
  if (value.trim().length === 0) {
    throw new ValidationError(
      `Parameter "${paramName}" must be a non-empty string`,
      { paramName }
    );
  }
}

/**
 * Validates detection options
 */
export interface DetectionOptions {
  minConfidence?: number;
  inputSize?: number;
  maxResults?: number;
}

export function validateDetectionOptions(options: DetectionOptions): DetectionOptions {
  const validated: DetectionOptions = {};

  if (options.minConfidence !== undefined) {
    assertValidProbability(options.minConfidence, 'minConfidence');
    validated.minConfidence = options.minConfidence;
  }

  if (options.inputSize !== undefined) {
    assertPositiveNumber(options.inputSize, 'inputSize');
    if (options.inputSize < 32 || options.inputSize > 2048) {
      throw new ValidationError(
        `Parameter "inputSize" must be between 32 and 2048, received: ${options.inputSize}`,
        { inputSize: options.inputSize }
      );
    }
    validated.inputSize = options.inputSize;
  }

  if (options.maxResults !== undefined) {
    assertPositiveNumber(options.maxResults, 'maxResults');
    if (!Number.isInteger(options.maxResults)) {
      throw new ValidationError(
        `Parameter "maxResults" must be an integer, received: ${options.maxResults}`,
        { maxResults: options.maxResults }
      );
    }
    validated.maxResults = options.maxResults;
  }

  return validated;
}

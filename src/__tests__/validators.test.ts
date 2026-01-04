import { describe, it, expect } from 'vitest';
import {
  assertValidNumber,
  assertPositiveNumber,
  assertNonNegativeNumber,
  assertValidProbability,
  assertValidDimensions,
  assertNonEmptyArray,
  assertArraysEqualLength,
  assertDefined,
  assertNonEmptyString,
  validateDetectionOptions,
} from '../validation';
import { ValidationError, InvalidDimensionsError } from '../errors';

describe('assertValidNumber', () => {
  it('should accept valid numbers', () => {
    expect(() => assertValidNumber(0, 'test')).not.toThrow();
    expect(() => assertValidNumber(42, 'test')).not.toThrow();
    expect(() => assertValidNumber(-10, 'test')).not.toThrow();
    expect(() => assertValidNumber(3.14, 'test')).not.toThrow();
  });

  it('should reject NaN', () => {
    expect(() => assertValidNumber(NaN, 'test')).toThrow(ValidationError);
  });

  it('should reject Infinity', () => {
    expect(() => assertValidNumber(Infinity, 'test')).toThrow(ValidationError);
    expect(() => assertValidNumber(-Infinity, 'test')).toThrow(ValidationError);
  });

  it('should reject non-numbers', () => {
    expect(() => assertValidNumber('42', 'test')).toThrow(ValidationError);
    expect(() => assertValidNumber(null, 'test')).toThrow(ValidationError);
    expect(() => assertValidNumber(undefined, 'test')).toThrow(ValidationError);
  });
});

describe('assertPositiveNumber', () => {
  it('should accept positive numbers', () => {
    expect(() => assertPositiveNumber(1, 'test')).not.toThrow();
    expect(() => assertPositiveNumber(0.001, 'test')).not.toThrow();
  });

  it('should reject zero', () => {
    expect(() => assertPositiveNumber(0, 'test')).toThrow(ValidationError);
  });

  it('should reject negative numbers', () => {
    expect(() => assertPositiveNumber(-1, 'test')).toThrow(ValidationError);
  });
});

describe('assertNonNegativeNumber', () => {
  it('should accept zero and positive numbers', () => {
    expect(() => assertNonNegativeNumber(0, 'test')).not.toThrow();
    expect(() => assertNonNegativeNumber(100, 'test')).not.toThrow();
  });

  it('should reject negative numbers', () => {
    expect(() => assertNonNegativeNumber(-0.001, 'test')).toThrow(ValidationError);
  });
});

describe('assertValidProbability', () => {
  it('should accept values between 0 and 1', () => {
    expect(() => assertValidProbability(0, 'test')).not.toThrow();
    expect(() => assertValidProbability(0.5, 'test')).not.toThrow();
    expect(() => assertValidProbability(1, 'test')).not.toThrow();
  });

  it('should reject values outside 0-1 range', () => {
    expect(() => assertValidProbability(-0.1, 'test')).toThrow(ValidationError);
    expect(() => assertValidProbability(1.1, 'test')).toThrow(ValidationError);
  });
});

describe('assertValidDimensions', () => {
  it('should accept valid dimensions', () => {
    expect(() => assertValidDimensions(100, 100)).not.toThrow();
    expect(() => assertValidDimensions(1920, 1080)).not.toThrow();
  });

  it('should reject too small dimensions', () => {
    expect(() => assertValidDimensions(0, 100)).toThrow(InvalidDimensionsError);
    expect(() => assertValidDimensions(100, 0)).toThrow(InvalidDimensionsError);
  });

  it('should reject too large dimensions', () => {
    expect(() => assertValidDimensions(20000, 100)).toThrow(InvalidDimensionsError);
  });

  it('should respect custom config', () => {
    expect(() => assertValidDimensions(50, 50, { minWidth: 100, minHeight: 100 }))
      .toThrow(InvalidDimensionsError);
  });
});

describe('assertNonEmptyArray', () => {
  it('should accept non-empty arrays', () => {
    expect(() => assertNonEmptyArray([1, 2, 3], 'test')).not.toThrow();
    expect(() => assertNonEmptyArray(['a'], 'test')).not.toThrow();
  });

  it('should reject empty arrays', () => {
    expect(() => assertNonEmptyArray([], 'test')).toThrow(ValidationError);
  });

  it('should reject non-arrays', () => {
    expect(() => assertNonEmptyArray('string', 'test')).toThrow(ValidationError);
    expect(() => assertNonEmptyArray({}, 'test')).toThrow(ValidationError);
  });
});

describe('assertArraysEqualLength', () => {
  it('should accept arrays of equal length', () => {
    expect(() => assertArraysEqualLength([1, 2], [3, 4], 'a', 'b')).not.toThrow();
    expect(() => assertArraysEqualLength([], [], 'a', 'b')).not.toThrow();
  });

  it('should reject arrays of different length', () => {
    expect(() => assertArraysEqualLength([1], [1, 2], 'a', 'b')).toThrow(ValidationError);
  });
});

describe('assertDefined', () => {
  it('should accept defined values', () => {
    expect(() => assertDefined(0, 'test')).not.toThrow();
    expect(() => assertDefined('', 'test')).not.toThrow();
    expect(() => assertDefined(false, 'test')).not.toThrow();
  });

  it('should reject null', () => {
    expect(() => assertDefined(null, 'test')).toThrow(ValidationError);
  });

  it('should reject undefined', () => {
    expect(() => assertDefined(undefined, 'test')).toThrow(ValidationError);
  });
});

describe('assertNonEmptyString', () => {
  it('should accept non-empty strings', () => {
    expect(() => assertNonEmptyString('hello', 'test')).not.toThrow();
    expect(() => assertNonEmptyString(' a ', 'test')).not.toThrow();
  });

  it('should reject empty strings', () => {
    expect(() => assertNonEmptyString('', 'test')).toThrow(ValidationError);
    expect(() => assertNonEmptyString('   ', 'test')).toThrow(ValidationError);
  });

  it('should reject non-strings', () => {
    expect(() => assertNonEmptyString(123, 'test')).toThrow(ValidationError);
  });
});

describe('validateDetectionOptions', () => {
  it('should accept valid options', () => {
    const options = { minConfidence: 0.5, inputSize: 512, maxResults: 10 };
    const validated = validateDetectionOptions(options);

    expect(validated.minConfidence).toBe(0.5);
    expect(validated.inputSize).toBe(512);
    expect(validated.maxResults).toBe(10);
  });

  it('should accept partial options', () => {
    const options = { minConfidence: 0.8 };
    const validated = validateDetectionOptions(options);

    expect(validated.minConfidence).toBe(0.8);
    expect(validated.inputSize).toBeUndefined();
  });

  it('should reject invalid minConfidence', () => {
    expect(() => validateDetectionOptions({ minConfidence: 1.5 })).toThrow(ValidationError);
    expect(() => validateDetectionOptions({ minConfidence: -0.1 })).toThrow(ValidationError);
  });

  it('should reject invalid inputSize', () => {
    expect(() => validateDetectionOptions({ inputSize: 16 })).toThrow(ValidationError);
    expect(() => validateDetectionOptions({ inputSize: 5000 })).toThrow(ValidationError);
  });

  it('should reject non-integer maxResults', () => {
    expect(() => validateDetectionOptions({ maxResults: 5.5 })).toThrow(ValidationError);
  });
});

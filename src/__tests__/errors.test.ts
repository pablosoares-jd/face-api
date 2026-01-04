import { describe, it, expect } from 'vitest';
import {
  FaceApiError,
  ValidationError,
  ModelLoadError,
  InferenceError,
  UnsupportedInputError,
  InvalidDimensionsError,
  ModelNotLoadedError,
  TensorError,
} from '../errors';

describe('FaceApiError', () => {
  it('should create error with message, code, and context', () => {
    const error = new FaceApiError('Test error', 'TEST_CODE', { key: 'value' });

    expect(error.message).toBe('Test error');
    expect(error.code).toBe('TEST_CODE');
    expect(error.context).toEqual({ key: 'value' });
    expect(error.name).toBe('FaceApiError');
    expect(error.timestamp).toBeInstanceOf(Date);
  });

  it('should serialize to JSON correctly', () => {
    const error = new FaceApiError('Test error', 'TEST_CODE', { foo: 'bar' });
    const json = error.toJSON();

    expect(json.name).toBe('FaceApiError');
    expect(json.message).toBe('Test error');
    expect(json.code).toBe('TEST_CODE');
    expect(json.context).toEqual({ foo: 'bar' });
    expect(typeof json.timestamp).toBe('string');
  });

  it('should be instanceof Error', () => {
    const error = new FaceApiError('Test', 'TEST');
    expect(error instanceof Error).toBe(true);
    expect(error instanceof FaceApiError).toBe(true);
  });
});

describe('ValidationError', () => {
  it('should create validation error with correct code', () => {
    const error = new ValidationError('Invalid input', { field: 'name' });

    expect(error.code).toBe('VALIDATION_ERROR');
    expect(error.name).toBe('ValidationError');
    expect(error.context).toEqual({ field: 'name' });
  });

  it('should be instanceof FaceApiError', () => {
    const error = new ValidationError('Test');
    expect(error instanceof FaceApiError).toBe(true);
  });
});

describe('ModelLoadError', () => {
  it('should create model load error with model info', () => {
    const error = new ModelLoadError('ssdMobilenetv1', 'Failed to load weights', '/path/to/model');

    expect(error.code).toBe('MODEL_LOAD_ERROR');
    expect(error.name).toBe('ModelLoadError');
    expect(error.modelName).toBe('ssdMobilenetv1');
    expect(error.modelPath).toBe('/path/to/model');
    expect(error.message).toBe('Failed to load weights');
  });
});

describe('InferenceError', () => {
  it('should create inference error with context', () => {
    const error = new InferenceError('Detection failed', { inputSize: 512 });

    expect(error.code).toBe('INFERENCE_ERROR');
    expect(error.name).toBe('InferenceError');
    expect(error.context).toEqual({ inputSize: 512 });
  });
});

describe('UnsupportedInputError', () => {
  it('should create error with input type info', () => {
    const supportedTypes = ['HTMLImageElement', 'HTMLCanvasElement', 'tf.Tensor3D'];
    const error = new UnsupportedInputError('Blob', supportedTypes);

    expect(error.code).toBe('UNSUPPORTED_INPUT_ERROR');
    expect(error.name).toBe('UnsupportedInputError');
    expect(error.inputType).toBe('Blob');
    expect(error.supportedTypes).toEqual(supportedTypes);
    expect(error.message).toContain('Blob');
    expect(error.message).toContain('HTMLImageElement');
  });
});

describe('InvalidDimensionsError', () => {
  it('should create error with dimension info', () => {
    const error = new InvalidDimensionsError(0, 100, 'Width must be positive');

    expect(error.code).toBe('INVALID_DIMENSIONS_ERROR');
    expect(error.name).toBe('InvalidDimensionsError');
    expect(error.message).toContain('0x100');
    expect(error.message).toContain('Width must be positive');
  });
});

describe('ModelNotLoadedError', () => {
  it('should create error with model name', () => {
    const error = new ModelNotLoadedError('faceRecognitionNet');

    expect(error.code).toBe('MODEL_NOT_LOADED_ERROR');
    expect(error.name).toBe('ModelNotLoadedError');
    expect(error.modelName).toBe('faceRecognitionNet');
    expect(error.message).toContain('faceRecognitionNet');
    expect(error.message).toContain('not loaded');
  });
});

describe('TensorError', () => {
  it('should create tensor error with context', () => {
    const error = new TensorError('Invalid tensor shape', { shape: [1, 2, 3] });

    expect(error.code).toBe('TENSOR_ERROR');
    expect(error.name).toBe('TensorError');
    expect(error.context).toEqual({ shape: [1, 2, 3] });
  });
});

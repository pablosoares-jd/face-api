/**
 * Base error class for all FaceAPI errors.
 * Provides consistent error handling with error codes and context.
 */
export class FaceApiError extends Error {
  public readonly code: string;
  public readonly context?: Record<string, unknown>;
  public readonly timestamp: Date;

  constructor(message: string, code: string, context?: Record<string, unknown>) {
    super(message);
    this.name = 'FaceApiError';
    this.code = code;
    this.context = context;
    this.timestamp = new Date();

    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, FaceApiError);
    }
  }

  public toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      context: this.context,
      timestamp: this.timestamp.toISOString(),
      stack: this.stack,
    };
  }
}

/**
 * Error thrown when input validation fails.
 */
export class ValidationError extends FaceApiError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'VALIDATION_ERROR', context);
    this.name = 'ValidationError';
  }
}

/**
 * Error thrown when model loading fails.
 */
export class ModelLoadError extends FaceApiError {
  public readonly modelName: string;
  public readonly modelPath?: string;

  constructor(modelName: string, message: string, modelPath?: string) {
    super(message, 'MODEL_LOAD_ERROR', { modelName, modelPath });
    this.name = 'ModelLoadError';
    this.modelName = modelName;
    this.modelPath = modelPath;
  }
}

/**
 * Error thrown when inference/detection fails.
 */
export class InferenceError extends FaceApiError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'INFERENCE_ERROR', context);
    this.name = 'InferenceError';
  }
}

/**
 * Error thrown when an unsupported input type is provided.
 */
export class UnsupportedInputError extends FaceApiError {
  public readonly inputType: string;
  public readonly supportedTypes: string[];

  constructor(inputType: string, supportedTypes: string[]) {
    super(
      `Unsupported input type: "${inputType}". Supported types: ${supportedTypes.join(', ')}`,
      'UNSUPPORTED_INPUT_ERROR',
      { inputType, supportedTypes },
    );
    this.name = 'UnsupportedInputError';
    this.inputType = inputType;
    this.supportedTypes = supportedTypes;
  }
}

/**
 * Error thrown when image dimensions are invalid.
 */
export class InvalidDimensionsError extends FaceApiError {
  constructor(width: number, height: number, reason: string) {
    super(
      `Invalid image dimensions: ${width}x${height}. ${reason}`,
      'INVALID_DIMENSIONS_ERROR',
      { width, height, reason },
    );
    this.name = 'InvalidDimensionsError';
  }
}

/**
 * Error thrown when a required model is not loaded.
 */
export class ModelNotLoadedError extends FaceApiError {
  public readonly modelName: string;

  constructor(modelName: string) {
    super(
      `Model "${modelName}" is not loaded. Please load the model before using it.`,
      'MODEL_NOT_LOADED_ERROR',
      { modelName },
    );
    this.name = 'ModelNotLoadedError';
    this.modelName = modelName;
  }
}

/**
 * Error thrown when tensor operations fail.
 */
export class TensorError extends FaceApiError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'TENSOR_ERROR', context);
    this.name = 'TensorError';
  }
}

import * as tf from '@tensorflow/tfjs';

/**
 * Memory profile for a neural network model.
 */
export interface ModelMemoryProfile {
  /** Model name */
  name: string;
  /** Approximate weight size in MB */
  weightSizeMB: number;
  /** Typical input tensor size in MB (for default input size) */
  inputSizeMB: number;
  /** Estimated peak memory during inference in MB */
  peakMemoryMB: number;
  /** Default input dimensions */
  defaultInputSize: number;
  /** Output dimensions */
  outputDimensions: string;
  /** Recommended batch size for different memory constraints */
  recommendedBatchSizes: {
    /** Low memory devices (<2GB VRAM) */
    lowMemory: number;
    /** Medium memory devices (2-4GB VRAM) */
    mediumMemory: number;
    /** High memory devices (>4GB VRAM) */
    highMemory: number;
  };
}

/**
 * Memory profiles for all face-api models.
 * Values are approximate and may vary based on backend and precision.
 */
export const MODEL_MEMORY_PROFILES: Record<string, ModelMemoryProfile> = {
  SsdMobilenetv1: {
    name: 'SSD MobileNetv1',
    weightSizeMB: 5.4,
    inputSizeMB: 1.0, // 512x512x3 float32
    peakMemoryMB: 25,
    defaultInputSize: 512,
    outputDimensions: 'boxes: [N, 4], scores: [N]',
    recommendedBatchSizes: {
      lowMemory: 1,
      mediumMemory: 2,
      highMemory: 4,
    },
  },

  BlazeFace: {
    name: 'BlazeFace',
    weightSizeMB: 0.4,
    inputSizeMB: 0.2, // 128x128x3 float32
    peakMemoryMB: 5,
    defaultInputSize: 128,
    outputDimensions: 'boxes: [N, 16], scores: [N]',
    recommendedBatchSizes: {
      lowMemory: 4,
      mediumMemory: 8,
      highMemory: 16,
    },
  },

  FaceRecognitionNet: {
    name: 'FaceRecognitionNet (FaceNet)',
    weightSizeMB: 6.2,
    inputSizeMB: 0.26, // 150x150x3 float32
    peakMemoryMB: 30,
    defaultInputSize: 150,
    outputDimensions: 'descriptor: [128]',
    recommendedBatchSizes: {
      lowMemory: 1,
      mediumMemory: 4,
      highMemory: 8,
    },
  },

  AdaFace: {
    name: 'AdaFace',
    weightSizeMB: 24,
    inputSizeMB: 0.15, // 112x112x3 float32
    peakMemoryMB: 80,
    defaultInputSize: 112,
    outputDimensions: 'descriptor: [512]',
    recommendedBatchSizes: {
      lowMemory: 1,
      mediumMemory: 2,
      highMemory: 4,
    },
  },

  FaceLandmark68Net: {
    name: 'FaceLandmark68Net',
    weightSizeMB: 0.35,
    inputSizeMB: 0.15, // 112x112x3 float32
    peakMemoryMB: 8,
    defaultInputSize: 112,
    outputDimensions: 'landmarks: [68, 2]',
    recommendedBatchSizes: {
      lowMemory: 4,
      mediumMemory: 8,
      highMemory: 16,
    },
  },

  FaceLandmark68TinyNet: {
    name: 'FaceLandmark68TinyNet',
    weightSizeMB: 0.08,
    inputSizeMB: 0.15, // 112x112x3 float32
    peakMemoryMB: 4,
    defaultInputSize: 112,
    outputDimensions: 'landmarks: [68, 2]',
    recommendedBatchSizes: {
      lowMemory: 8,
      mediumMemory: 16,
      highMemory: 32,
    },
  },

  AgeGenderNet: {
    name: 'AgeGenderNet',
    weightSizeMB: 0.42,
    inputSizeMB: 0.15, // 112x112x3 float32
    peakMemoryMB: 10,
    defaultInputSize: 112,
    outputDimensions: 'age: [1], gender: [2]',
    recommendedBatchSizes: {
      lowMemory: 4,
      mediumMemory: 8,
      highMemory: 16,
    },
  },

  FaceExpressionNet: {
    name: 'FaceExpressionNet',
    weightSizeMB: 0.31,
    inputSizeMB: 0.15, // 112x112x3 float32
    peakMemoryMB: 8,
    defaultInputSize: 112,
    outputDimensions: 'expressions: [7]',
    recommendedBatchSizes: {
      lowMemory: 4,
      mediumMemory: 8,
      highMemory: 16,
    },
  },

  TinyFaceDetector: {
    name: 'TinyFaceDetector',
    weightSizeMB: 0.19,
    inputSizeMB: 0.4, // 416x416x3 float32 (varies with inputSize option)
    peakMemoryMB: 15,
    defaultInputSize: 416,
    outputDimensions: 'boxes: [N, 4], scores: [N]',
    recommendedBatchSizes: {
      lowMemory: 1,
      mediumMemory: 2,
      highMemory: 4,
    },
  },

  FaceMesh: {
    name: 'FaceMesh (MediaPipe)',
    weightSizeMB: 2.5,
    inputSizeMB: 0.4, // 192x192x3 float32
    peakMemoryMB: 40,
    defaultInputSize: 192,
    outputDimensions: 'landmarks: [468, 3] or [478, 3]',
    recommendedBatchSizes: {
      lowMemory: 1,
      mediumMemory: 2,
      highMemory: 4,
    },
  },
};

/**
 * Get memory profile for a model.
 */
export function getModelMemoryProfile(modelName: string): ModelMemoryProfile | undefined {
  return MODEL_MEMORY_PROFILES[modelName];
}

/**
 * Get current TensorFlow.js memory info.
 */
export function getMemoryInfo(): tf.MemoryInfo {
  return tf.memory();
}

/**
 * Estimate total memory required for a pipeline.
 * @param modelNames Array of model names to use
 * @param batchSize Batch size for inference
 * @returns Estimated memory in MB
 */
export function estimatePipelineMemory(modelNames: string[], batchSize = 1): number {
  let totalWeights = 0;
  let maxPeakMemory = 0;

  for (const name of modelNames) {
    const profile = MODEL_MEMORY_PROFILES[name];
    if (profile) {
      totalWeights += profile.weightSizeMB;
      const peakWithBatch = profile.peakMemoryMB + (profile.inputSizeMB * (batchSize - 1));
      maxPeakMemory = Math.max(maxPeakMemory, peakWithBatch);
    }
  }

  // Total = weights (always loaded) + max peak during inference
  return totalWeights + maxPeakMemory;
}

/**
 * Check if a pipeline will fit in available memory.
 * @param modelNames Array of model names
 * @param availableMemoryMB Available GPU/system memory in MB
 * @param batchSize Desired batch size
 * @returns Whether the pipeline will fit and recommendations
 */
export function checkMemoryFit(
  modelNames: string[],
  availableMemoryMB: number,
  batchSize = 1,
): {
  fits: boolean;
  estimatedUsageMB: number;
  recommendedBatchSize: number;
  warnings: string[];
} {
  const estimatedUsage = estimatePipelineMemory(modelNames, batchSize);
  const fits = estimatedUsage < availableMemoryMB * 0.8; // 80% threshold

  // Find max recommended batch size
  let recommendedBatchSize = batchSize;
  const warnings: string[] = [];

  if (!fits) {
    // Try to find a batch size that fits
    for (let bs = batchSize - 1; bs >= 1; bs--) {
      const usage = estimatePipelineMemory(modelNames, bs);
      if (usage < availableMemoryMB * 0.8) {
        recommendedBatchSize = bs;
        break;
      }
    }

    warnings.push(
      `Estimated memory usage (${estimatedUsage.toFixed(1)}MB) exceeds 80% of available memory (${availableMemoryMB}MB)`,
    );

    if (recommendedBatchSize < batchSize) {
      warnings.push(`Consider reducing batch size to ${recommendedBatchSize}`);
    }
  }

  // Check for heavy models
  for (const name of modelNames) {
    const profile = MODEL_MEMORY_PROFILES[name];
    if (profile && profile.peakMemoryMB > 50) {
      warnings.push(`${profile.name} is memory-intensive (peak: ${profile.peakMemoryMB}MB)`);
    }
  }

  return {
    fits,
    estimatedUsageMB: estimatedUsage,
    recommendedBatchSize,
    warnings,
  };
}

/**
 * Memory monitoring utilities.
 */
export const memoryMonitor = {
  /**
   * Log current memory state.
   */
  logMemoryState(label = 'Memory'): void {
    const info = tf.memory();
    console.info(`[${label}] Tensors: ${info.numTensors}, Bytes: ${(info.numBytes / 1024 / 1024).toFixed(2)}MB`);
  },

  /**
   * Run a function and report memory delta.
   */
  async trackMemory<T>(fn: () => Promise<T>, label = 'Operation'): Promise<{ result: T; memoryDelta: number }> {
    const before = tf.memory();
    const result = await fn();
    const after = tf.memory();

    const delta = (after.numBytes - before.numBytes) / 1024 / 1024;
    console.info(`[${label}] Memory delta: ${delta.toFixed(2)}MB (Tensors: ${before.numTensors} -> ${after.numTensors})`);

    return { result, memoryDelta: delta };
  },

  /**
   * Check for potential memory leaks.
   * Call at start and end of operations to verify tensors are properly disposed.
   */
  createLeakChecker(): { check: () => { leaked: boolean; tensorDelta: number; byteDelta: number } } {
    const initial = tf.memory();

    return {
      check: () => {
        const current = tf.memory();
        const tensorDelta = current.numTensors - initial.numTensors;
        const byteDelta = current.numBytes - initial.numBytes;

        return {
          leaked: tensorDelta > 0,
          tensorDelta,
          byteDelta,
        };
      },
    };
  },
};

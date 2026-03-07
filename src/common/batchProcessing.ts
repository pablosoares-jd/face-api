import * as tf from '@tensorflow/tfjs';

/**
 * Options for batch processing.
 */
export interface BatchProcessingOptions {
  /** Maximum batch size per inference call. Default: 4 */
  batchSize?: number;
  /** Whether to process batches in parallel. Default: false (sequential is more memory efficient) */
  parallel?: boolean;
  /** Callback for progress updates */
  onProgress?: (completed: number, total: number) => void;
  /** Whether to dispose intermediate tensors aggressively. Default: true */
  aggressiveDispose?: boolean;
}

/**
 * Result of batch processing.
 */
export interface BatchResult<T> {
  /** Results for each input */
  results: T[];
  /** Processing time in milliseconds */
  processingTimeMs: number;
  /** Number of batches processed */
  batchCount: number;
}

/**
 * Split an array into chunks of specified size.
 */
export function chunkArray<T>(array: T[], chunkSize: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    chunks.push(array.slice(i, i + chunkSize));
  }
  return chunks;
}

/**
 * Process items in batches with a given processor function.
 * Handles memory management and provides progress updates.
 *
 * @param items Items to process
 * @param processor Function to process a batch of items
 * @param options Batch processing options
 * @returns Batch processing result
 *
 * @example
 * ```typescript
 * const results = await processBatches(
 *   images,
 *   async (batch) => {
 *     return await model.detect(batch);
 *   },
 *   { batchSize: 4, onProgress: (done, total) => console.log(`${done}/${total}`) }
 * );
 * ```
 */
export async function processBatches<T, R>(
  items: T[],
  processor: (batch: T[]) => Promise<R[]>,
  options: BatchProcessingOptions = {},
): Promise<BatchResult<R>> {
  const {
    batchSize = 4,
    parallel = false,
    onProgress,
    aggressiveDispose = true,
  } = options;

  const startTime = performance.now();
  const chunks = chunkArray(items, batchSize);
  const allResults: R[] = [];

  if (parallel) {
    // Process all batches in parallel (higher memory usage)
    const batchPromises = chunks.map(async (chunk, idx) => {
      const results = await processor(chunk);

      if (aggressiveDispose) {
        // Run garbage collection hint between batches
        await tf.nextFrame();
      }

      if (onProgress) {
        onProgress((idx + 1) * batchSize, items.length);
      }

      return results;
    });

    const batchResults = await Promise.all(batchPromises);
    for (const results of batchResults) {
      allResults.push(...results);
    }
  } else {
    // Process batches sequentially (lower memory usage)
    let completed = 0;
    for (const chunk of chunks) {
      const results = await processor(chunk);
      allResults.push(...results);

      completed += chunk.length;

      if (aggressiveDispose) {
        // Allow GPU to clean up between batches
        await tf.nextFrame();
      }

      if (onProgress) {
        onProgress(completed, items.length);
      }
    }
  }

  const endTime = performance.now();

  return {
    results: allResults,
    processingTimeMs: endTime - startTime,
    batchCount: chunks.length,
  };
}

/**
 * Create a batched tensor from multiple tensors.
 * Handles different input sizes by resizing to target size.
 *
 * @param tensors Array of 3D tensors (HWC format)
 * @param targetSize Target size for width and height
 * @param maintainAspectRatio Whether to maintain aspect ratio (pad with zeros)
 * @returns Batched 4D tensor (NHWC format)
 */
export function createBatchTensor(
  tensors: tf.Tensor3D[],
  targetSize: number,
  maintainAspectRatio = true,
): tf.Tensor4D {
  return tf.tidy(() => {
    const resized = tensors.map((t) => {
      if (maintainAspectRatio) {
        return tf.image.resizeBilinear(
          t.expandDims(0) as tf.Tensor4D,
          [targetSize, targetSize],
        ).squeeze([0]) as tf.Tensor3D;
      }
      return tf.image.resizeBilinear(
          t.expandDims(0) as tf.Tensor4D,
          [targetSize, targetSize],
      ).squeeze([0]) as tf.Tensor3D;
    });

    return tf.stack(resized) as tf.Tensor4D;
  });
}

/**
 * Unbatch a 4D tensor result into individual results.
 *
 * @param batchedResult The batched result tensor
 * @returns Array of individual result tensors
 */
export function unbatchTensor<T extends tf.Tensor>(
  batchedResult: tf.Tensor,
): T[] {
  return tf.unstack(batchedResult) as T[];
}

/**
 * Process face detection results in batch.
 * Optimized for detecting faces in multiple images simultaneously.
 *
 * @param forwardFn The forward function of the detection model
 * @param inputs Array of input tensors
 * @param postProcessFn Post-processing function for each batch result
 * @returns Array of post-processed results
 */
export async function batchDetection<TInput, TOutput>(
  forwardFn: (input: TInput) => tf.Tensor | tf.Tensor[],
  inputs: TInput[],
  postProcessFn: (output: tf.Tensor | tf.Tensor[], inputIdx: number) => Promise<TOutput>,
): Promise<TOutput[]> {
  const results: TOutput[] = [];

  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];
    if (input === undefined) continue;
    const output = forwardFn(input);

    try {
      const result = await postProcessFn(output, i);
      results.push(result);
    } finally {
      // Dispose output tensors
      if (Array.isArray(output)) {
        output.forEach((t) => t.dispose());
      } else if (output instanceof tf.Tensor) {
        output.dispose();
      }
    }

    // Allow GPU cleanup
    await tf.nextFrame();
  }

  return results;
}

/**
 * Parallel async map with concurrency limit.
 * Useful for processing items with external async operations.
 *
 * @param items Items to process
 * @param mapper Async mapping function
 * @param concurrency Maximum number of concurrent operations
 * @returns Mapped results in original order
 */
export async function parallelMap<T, R>(
  items: T[],
  mapper: (item: T, index: number) => Promise<R>,
  concurrency: number,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let currentIndex = 0;

  async function worker(): Promise<void> {
    while (currentIndex < items.length) {
      const index = currentIndex++;
      const item = items[index];
      if (item !== undefined) {
        results[index] = await mapper(item, index);
      }
    }
  }

  // Start workers up to concurrency limit
  const workers = Array(Math.min(concurrency, items.length))
    .fill(null)
    .map(() => worker());

  await Promise.all(workers);

  return results;
}

/**
 * Batch descriptor computation for multiple face crops.
 * Optimized for computing face embeddings/descriptors in batch.
 *
 * @param computeDescriptor Function to compute descriptor for a single input
 * @param inputs Array of face crop tensors/images
 * @param batchSize Number of faces to process per batch
 * @returns Array of face descriptors
 */
export async function batchComputeDescriptors(
  computeDescriptor: (input: unknown) => Promise<Float32Array>,
  inputs: unknown[],
  batchSize = 4,
): Promise<Float32Array[]> {
  const result = await processBatches(
    inputs,
    async (batch) => {
      const descriptors = await Promise.all(
        batch.map((input) => computeDescriptor(input)),
      );
      return descriptors;
    },
    { batchSize },
  );

  return result.results;
}

/**
 * Estimate optimal batch size based on available memory and model profile.
 *
 * @param modelName Name of the model
 * @param availableMemoryMB Available GPU memory in MB
 * @returns Recommended batch size
 */
export function estimateOptimalBatchSize(
  modelName: string,
  availableMemoryMB: number,
): number {
  // Import would create circular dependency, so use hardcoded estimates
  const modelMemoryEstimates: Record<string, { weight: number; perInput: number }> = {
    SsdMobilenetv1: { weight: 5.4, perInput: 1.0 },
    BlazeFace: { weight: 0.4, perInput: 0.2 },
    FaceRecognitionNet: { weight: 6.2, perInput: 0.26 },
    AdaFace: { weight: 24, perInput: 0.15 },
    FaceLandmark68Net: { weight: 0.35, perInput: 0.15 },
    AgeGenderNet: { weight: 0.42, perInput: 0.15 },
    FaceExpressionNet: { weight: 0.31, perInput: 0.15 },
  };

  const estimate = modelMemoryEstimates[modelName];
  if (!estimate) {
    return 4; // Default
  }

  // Reserve 80% of available memory, subtract weight size
  const usableMemory = availableMemoryMB * 0.8 - estimate.weight;

  // Calculate batch size based on per-input memory
  // Account for peak memory being ~2-3x input size during inference
  const batchSize = Math.floor(usableMemory / (estimate.perInput * 3));

  return Math.max(1, Math.min(16, batchSize));
}

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
export declare function chunkArray<T>(array: T[], chunkSize: number): T[][];
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
export declare function processBatches<T, R>(items: T[], processor: (batch: T[]) => Promise<R[]>, options?: BatchProcessingOptions): Promise<BatchResult<R>>;
/**
 * Create a batched tensor from multiple tensors.
 * Handles different input sizes by resizing to target size.
 *
 * @param tensors Array of 3D tensors (HWC format)
 * @param targetSize Target size for width and height
 * @param maintainAspectRatio Whether to maintain aspect ratio (pad with zeros)
 * @returns Batched 4D tensor (NHWC format)
 */
export declare function createBatchTensor(tensors: tf.Tensor3D[], targetSize: number, maintainAspectRatio?: boolean): tf.Tensor4D;
/**
 * Unbatch a 4D tensor result into individual results.
 *
 * @param batchedResult The batched result tensor
 * @returns Array of individual result tensors
 */
export declare function unbatchTensor<T extends tf.Tensor>(batchedResult: tf.Tensor): T[];
/**
 * Process face detection results in batch.
 * Optimized for detecting faces in multiple images simultaneously.
 *
 * @param forwardFn The forward function of the detection model
 * @param inputs Array of input tensors
 * @param postProcessFn Post-processing function for each batch result
 * @returns Array of post-processed results
 */
export declare function batchDetection<TInput, TOutput>(forwardFn: (input: TInput) => tf.Tensor | tf.Tensor[], inputs: TInput[], postProcessFn: (output: tf.Tensor | tf.Tensor[], inputIdx: number) => Promise<TOutput>): Promise<TOutput[]>;
/**
 * Parallel async map with concurrency limit.
 * Useful for processing items with external async operations.
 *
 * @param items Items to process
 * @param mapper Async mapping function
 * @param concurrency Maximum number of concurrent operations
 * @returns Mapped results in original order
 */
export declare function parallelMap<T, R>(items: T[], mapper: (item: T, index: number) => Promise<R>, concurrency: number): Promise<R[]>;
/**
 * Batch descriptor computation for multiple face crops.
 * Optimized for computing face embeddings/descriptors in batch.
 *
 * @param computeDescriptor Function to compute descriptor for a single input
 * @param inputs Array of face crop tensors/images
 * @param batchSize Number of faces to process per batch
 * @returns Array of face descriptors
 */
export declare function batchComputeDescriptors(computeDescriptor: (input: unknown) => Promise<Float32Array>, inputs: unknown[], batchSize?: number): Promise<Float32Array[]>;
/**
 * Estimate optimal batch size based on available memory and model profile.
 *
 * @param modelName Name of the model
 * @param availableMemoryMB Available GPU memory in MB
 * @returns Recommended batch size
 */
export declare function estimateOptimalBatchSize(modelName: string, availableMemoryMB: number): number;

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
export declare const MODEL_MEMORY_PROFILES: Record<string, ModelMemoryProfile>;
/**
 * Get memory profile for a model.
 */
export declare function getModelMemoryProfile(modelName: string): ModelMemoryProfile | undefined;
/**
 * Get current TensorFlow.js memory info.
 */
export declare function getMemoryInfo(): tf.MemoryInfo;
/**
 * Estimate total memory required for a pipeline.
 * @param modelNames Array of model names to use
 * @param batchSize Batch size for inference
 * @returns Estimated memory in MB
 */
export declare function estimatePipelineMemory(modelNames: string[], batchSize?: number): number;
/**
 * Check if a pipeline will fit in available memory.
 * @param modelNames Array of model names
 * @param availableMemoryMB Available GPU/system memory in MB
 * @param batchSize Desired batch size
 * @returns Whether the pipeline will fit and recommendations
 */
export declare function checkMemoryFit(modelNames: string[], availableMemoryMB: number, batchSize?: number): {
    fits: boolean;
    estimatedUsageMB: number;
    recommendedBatchSize: number;
    warnings: string[];
};
/**
 * Memory monitoring utilities.
 */
export declare const memoryMonitor: {
    /**
     * Log current memory state.
     */
    logMemoryState(label?: string): void;
    /**
     * Run a function and report memory delta.
     */
    trackMemory<T>(fn: () => Promise<T>, label?: string): Promise<{
        result: T;
        memoryDelta: number;
    }>;
    /**
     * Check for potential memory leaks.
     * Call at start and end of operations to verify tensors are properly disposed.
     */
    createLeakChecker(): {
        check: () => {
            leaked: boolean;
            tensorDelta: number;
            byteDelta: number;
        };
    };
};

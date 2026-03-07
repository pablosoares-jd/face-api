/**
 * WebGPU backend initialization and utilities.
 * Provides helpers for setting up WebGPU with face-api.js.
 */
export interface WebGPUInfo {
    supported: boolean;
    adapter: GPUAdapter | null;
    device: GPUDevice | null;
    adapterInfo: GPUAdapterInfo | null;
}
/**
 * Check if WebGPU is supported in the current environment.
 * WebGPU requires Chrome 113+, Edge 113+, or compatible browsers.
 */
export declare function isWebGPUSupported(): boolean;
/**
 * Get detailed WebGPU support information.
 * Returns adapter and device info if available.
 */
export declare function getWebGPUInfo(): Promise<WebGPUInfo>;
/**
 * Initialize WebGPU backend for TensorFlow.js.
 * This must be called before using any face-api.js functions.
 *
 * @throws Error if WebGPU is not supported
 * @returns Promise that resolves when backend is ready
 *
 * @example
 * ```typescript
 * import * as faceapi from '@vladmandic/face-api/webgpu';
 * import { initWebGPU } from '@vladmandic/face-api/webgpu';
 *
 * await initWebGPU();
 * // Now use face-api normally
 * const detections = await faceapi.detectAllFaces(image);
 * ```
 */
export declare function initWebGPU(): Promise<void>;
/**
 * Get current TensorFlow.js backend name.
 */
export declare function getCurrentBackend(): string;
/**
 * Check if WebGPU backend is currently active.
 */
export declare function isWebGPUActive(): boolean;
/**
 * Backend priority order for automatic fallback.
 */
export type BackendType = 'webgpu' | 'webgl' | 'wasm' | 'cpu';
/**
 * Backend performance characteristics.
 */
export interface BackendPerformance {
    backend: BackendType;
    estimatedSpeedup: number;
    memoryEfficiency: 'high' | 'medium' | 'low';
    parallelization: 'gpu' | 'simd' | 'none';
}
/**
 * Get performance characteristics for each backend.
 */
export declare function getBackendPerformance(): BackendPerformance[];
/**
 * Result of backend initialization with fallback.
 */
export interface BackendInitResult {
    /** The backend that was successfully initialized */
    backend: BackendType;
    /** Whether this was the preferred backend */
    isPreferred: boolean;
    /** Error message if preferred backend failed */
    fallbackReason?: string;
}
/**
 * Initialize the best available backend with automatic fallback.
 * Tries WebGPU first, falls back to WebGL, then CPU.
 *
 * @param preferredBackend - Preferred backend to try first (default: 'webgpu')
 * @returns Result indicating which backend was initialized
 *
 * @example
 * ```typescript
 * const result = await initBestBackend();
 * console.log(`Using ${result.backend} backend`);
 * if (!result.isPreferred) {
 *   console.log(`Fallback reason: ${result.fallbackReason}`);
 * }
 * ```
 */
/**
 * Check if WASM backend is available.
 */
export declare function isWasmSupported(): boolean;
export declare function initBestBackend(preferredBackend?: BackendType): Promise<BackendInitResult>;
/**
 * Initialize WebGPU with automatic fallback to WebGL/CPU.
 * Unlike initWebGPU(), this never throws and always succeeds.
 *
 * @returns Result indicating which backend was initialized
 *
 * @example
 * ```typescript
 * const { backend, isPreferred } = await initWebGPUWithFallback();
 * if (backend === 'webgpu') {
 *   console.log('🚀 WebGPU enabled - maximum performance!');
 * } else {
 *   console.log(`Using ${backend} fallback`);
 * }
 * ```
 */
export declare function initWebGPUWithFallback(): Promise<BackendInitResult>;
/**
 * Check if WebGL is supported.
 */
export declare function isWebGLSupported(): boolean;
/**
 * Get information about all available backends.
 */
export declare function getAvailableBackends(): {
    backend: BackendType;
    supported: boolean;
}[];
export declare function ensureBackendInitialized(): Promise<BackendInitResult>;
/**
 * Reset backend state (for testing).
 */
export declare function resetBackendState(): void;
/**
 * Recommended way to initialize face-api.
 * Automatically selects the best backend with WebGPU priority.
 *
 * @example
 * ```typescript
 * import * as faceapi from '@vladmandic/face-api';
 *
 * // Initialize with best available backend
 * await faceapi.init();
 *
 * // Or with specific preference
 * await faceapi.init({ preferredBackend: 'webgl' });
 * ```
 */
export declare function init(options?: {
    preferredBackend?: BackendType;
    silent?: boolean;
}): Promise<BackendInitResult>;

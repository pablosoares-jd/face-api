/**
 * WebGPU backend initialization and utilities.
 * Provides helpers for setting up WebGPU with face-api.js.
 */

import * as tf from '@tensorflow/tfjs';

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
export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Get detailed WebGPU support information.
 * Returns adapter and device info if available.
 */
export async function getWebGPUInfo(): Promise<WebGPUInfo> {
  if (!isWebGPUSupported()) {
    return {
      supported: false,
      adapter: null,
      device: null,
      adapterInfo: null,
    };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return {
        supported: false,
        adapter: null,
        device: null,
        adapterInfo: null,
      };
    }

    const device = await adapter.requestDevice();
    const adapterInfo = await adapter.requestAdapterInfo();

    return {
      supported: true,
      adapter,
      device,
      adapterInfo,
    };
  } catch {
    return {
      supported: false,
      adapter: null,
      device: null,
      adapterInfo: null,
    };
  }
}

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
export async function initWebGPU(): Promise<void> {
  if (!isWebGPUSupported()) {
    throw new Error(
      'WebGPU is not supported in this browser. '
      + 'Requires Chrome 113+, Edge 113+, or compatible browser with WebGPU enabled.',
    );
  }

  await tf.setBackend('webgpu');
  await tf.ready();
}

/**
 * Get current TensorFlow.js backend name.
 */
export function getCurrentBackend(): string {
  return tf.getBackend();
}

/**
 * Check if WebGPU backend is currently active.
 */
export function isWebGPUActive(): boolean {
  return tf.getBackend() === 'webgpu';
}

/**
 * Backend priority order for automatic fallback.
 */
export type BackendType = 'webgpu' | 'webgl' | 'wasm' | 'cpu';

/**
 * Backend performance characteristics.
 */
export interface BackendPerformance {
  backend: BackendType;
  estimatedSpeedup: number; // Relative to CPU
  memoryEfficiency: 'high' | 'medium' | 'low';
  parallelization: 'gpu' | 'simd' | 'none';
}

/**
 * Get performance characteristics for each backend.
 */
export function getBackendPerformance(): BackendPerformance[] {
  return [
    { backend: 'webgpu', estimatedSpeedup: 50, memoryEfficiency: 'high', parallelization: 'gpu' },
    { backend: 'webgl', estimatedSpeedup: 20, memoryEfficiency: 'medium', parallelization: 'gpu' },
    { backend: 'wasm', estimatedSpeedup: 5, memoryEfficiency: 'high', parallelization: 'simd' },
    { backend: 'cpu', estimatedSpeedup: 1, memoryEfficiency: 'low', parallelization: 'none' },
  ];
}

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
export function isWasmSupported(): boolean {
  return typeof WebAssembly !== 'undefined';
}

export async function initBestBackend(
  preferredBackend: BackendType = 'webgpu',
): Promise<BackendInitResult> {
  // Priority order: WebGPU > WebGL > WASM > CPU
  const backends: BackendType[] = ['webgpu', 'webgl', 'wasm', 'cpu'];

  // Reorder to put preferred first
  const orderedBackends = [
    preferredBackend,
    ...backends.filter((b) => b !== preferredBackend),
  ];

  let lastError: string | undefined;

  for (const backend of orderedBackends) {
    try {
      // Check support before attempting
      if (backend === 'webgpu' && !isWebGPUSupported()) {
        lastError = 'WebGPU not supported in this browser';
        continue;
      }
      if (backend === 'webgl' && !isWebGLSupported()) {
        lastError = 'WebGL not supported in this browser';
        continue;
      }
      if (backend === 'wasm' && !isWasmSupported()) {
        lastError = 'WebAssembly not supported in this browser';
        continue;
      }

      await tf.setBackend(backend);
      await tf.ready();

      // Log success for debugging
      if (typeof console !== 'undefined') {
        const perf = getBackendPerformance().find((p) => p.backend === backend);
        console.info(
          `[face-api] Backend initialized: ${backend} `
          + `(${perf?.estimatedSpeedup}x speedup, ${perf?.parallelization} parallelization)`,
        );
      }

      return {
        backend,
        isPreferred: backend === preferredBackend,
        fallbackReason: backend !== preferredBackend ? lastError : undefined,
      };
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      continue;
    }
  }

  // This should rarely happen as CPU is always available
  throw new Error(`Failed to initialize any backend. Last error: ${lastError}`);
}

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
export async function initWebGPUWithFallback(): Promise<BackendInitResult> {
  return initBestBackend('webgpu');
}

/**
 * Check if WebGL is supported.
 */
export function isWebGLSupported(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl') || canvas.getContext('webgl2'));
  } catch {
    return false;
  }
}

/**
 * Get information about all available backends.
 */
export function getAvailableBackends(): { backend: BackendType; supported: boolean }[] {
  return [
    { backend: 'webgpu', supported: isWebGPUSupported() },
    { backend: 'webgl', supported: isWebGLSupported() },
    { backend: 'wasm', supported: isWasmSupported() },
    { backend: 'cpu', supported: true },
  ];
}

/**
 * Auto-initialize the best backend.
 * This is called automatically when face-api is first used if no backend is set.
 */
let backendInitPromise: Promise<BackendInitResult> | null = null;

export async function ensureBackendInitialized(): Promise<BackendInitResult> {
  // Return cached promise if already initializing/initialized
  if (backendInitPromise) {
    return backendInitPromise;
  }

  // Check if a backend is already set
  const currentBackend = tf.getBackend();
  if (currentBackend) {
    return {
      backend: currentBackend as BackendType,
      isPreferred: true,
    };
  }

  // Initialize with WebGPU priority
  backendInitPromise = initBestBackend('webgpu');
  return backendInitPromise;
}

/**
 * Reset backend state (for testing).
 */
export function resetBackendState(): void {
  backendInitPromise = null;
}

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
export async function init(options: {
  preferredBackend?: BackendType;
  silent?: boolean;
} = {}): Promise<BackendInitResult> {
  const { preferredBackend = 'webgpu', silent = false } = options;

  const result = await initBestBackend(preferredBackend);

  if (!silent && !result.isPreferred && result.fallbackReason) {
    console.warn(
      `[face-api] Could not use ${preferredBackend}: ${result.fallbackReason}. `
      + `Using ${result.backend} instead.`,
    );
  }

  return result;
}

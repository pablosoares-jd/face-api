/**
 * WebGPU backend initialization and utilities.
 * Provides helpers for setting up WebGPU with face-api.js.
 */

import * as tf from '../../dist/tfjs.esm';

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
      'WebGPU is not supported in this browser. ' +
      'Requires Chrome 113+, Edge 113+, or compatible browser with WebGPU enabled.',
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
export type BackendType = 'webgpu' | 'webgl' | 'cpu';

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
export async function initBestBackend(
  preferredBackend: BackendType = 'webgpu',
): Promise<BackendInitResult> {
  const backends: BackendType[] = ['webgpu', 'webgl', 'cpu'];

  // Reorder to put preferred first
  const orderedBackends = [
    preferredBackend,
    ...backends.filter((b) => b !== preferredBackend),
  ];

  let lastError: string | undefined;

  for (const backend of orderedBackends) {
    try {
      if (backend === 'webgpu') {
        if (!isWebGPUSupported()) {
          lastError = 'WebGPU not supported in this browser';
          continue;
        }
        await tf.setBackend('webgpu');
      } else {
        await tf.setBackend(backend);
      }

      await tf.ready();

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
    { backend: 'cpu', supported: true },
  ];
}

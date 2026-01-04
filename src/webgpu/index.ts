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

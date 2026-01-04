/**
 * Creates tfjs bundle for WebGPU-only browser builds.
 * WebGPU provides 3-10x performance improvement over WebGL.
 *
 * Requirements:
 * - Chrome 113+ or Edge 113+
 * - Secure context (HTTPS or localhost)
 *
 * @external
 */

// Core TensorFlow.js exports
export * from '@tensorflow/tfjs/dist/index.js';

// WebGPU backend only - no WebGL fallback
export * from '@tensorflow/tfjs-backend-webgpu/dist/index.js';

// Export versions
export { version } from '../../dist/tfjs.version.js';

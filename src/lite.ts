/**
 * Face-API Lite - Lightweight face detection only.
 *
 * Includes:
 * - SSD MobileNet v1 face detector
 * - TinyFaceDetector
 * - BlazeFace (modern, faster)
 *
 * Bundle size: ~500KB (vs ~1.2MB full)
 *
 * @example
 * ```typescript
 * import * as faceapi from '@vladmandic/face-api/lite';
 *
 * await faceapi.init();
 * const detections = await faceapi.detectAllFaces(image);
 * ```
 */

import * as tf from '@tensorflow/tfjs';

// Backend utilities
export {
  init,
  initBestBackend,
  initWebGPU,
  initWebGPUWithFallback,
  isWebGPUSupported,
  isWebGLSupported,
  isWasmSupported,
  getAvailableBackends,
  getCurrentBackend,
  BackendType,
  BackendInitResult,
} from './webgpu/index';

// Core classes
export {
  Box,
  BoundingBox,
  Dimensions,
  Point,
  Rect,
  FaceDetection,
  ObjectDetection,
} from './classes/index';

// DOM utilities
export {
  NetInput,
  TNetInput,
  toNetInput,
  createCanvas,
  createCanvasFromMedia,
  matchDimensions,
  fetchImage,
  fetchJson,
  fetchNetWeights,
  bufferToImage,
  imageTensorToCanvas,
  getContext2dOrThrow,
  getMediaDimensions,
  isMediaLoaded,
  awaitMediaLoaded,
  resolveInput,
} from './dom/index';

// Drawing utilities
export * as draw from './draw/index';

// Detectors
export { SsdMobilenetv1 } from './ssdMobilenetv1/SsdMobilenetv1';
export { SsdMobilenetv1Options, ISsdMobilenetv1Options } from './ssdMobilenetv1/SsdMobilenetv1Options';
export { TinyFaceDetector } from './tinyFaceDetector/TinyFaceDetector';
export { TinyFaceDetectorOptions, ITinyFaceDetectorOptions } from './tinyFaceDetector/TinyFaceDetectorOptions';
export { BlazeFace, BlazeFaceDetection, BlazeFaceKeypoints } from './blazeFace/BlazeFace';
export { BlazeFaceOptions, IBlazeFaceOptions } from './blazeFace/BlazeFaceOptions';

// Global detection API
export {
  detectSingleFace,
  detectAllFaces,
  DetectAllFacesTask,
  DetectSingleFaceTask,
} from './globalApi/index';

// NeuralNetwork base
export { NeuralNetwork } from './NeuralNetwork';

// Environment
export { env } from './env/index';

// Utilities
export * as utils from './utils/index';

// Resize helper
export { resizeResults } from './resizeResults';

// Export tf for advanced usage
export { tf };

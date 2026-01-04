/**
 * Face-API Full - Complete face analysis suite.
 *
 * Includes everything from lite plus:
 * - Face landmarks (68-point and FaceMesh 468-point)
 * - Face recognition (FaceNet and AdaFace)
 * - Age & gender prediction
 * - Expression detection
 * - KYC (Know Your Customer) verification
 *
 * @example
 * ```typescript
 * import * as faceapi from '@vladmandic/face-api/full';
 *
 * await faceapi.init();
 *
 * const result = await faceapi
 *   .detectSingleFace(image)
 *   .withFaceLandmarks()
 *   .withFaceDescriptor()
 *   .withAgeAndGender()
 *   .withFaceExpressions();
 * ```
 */

// Export everything from main index
export * from './index';

/**
 * Options for FaceMesh landmark detector.
 */
export interface IFaceMeshOptions {
  /**
   * Maximum number of faces to detect landmarks for.
   * Default: 1
   */
  maxFaces?: number;

  /**
   * Whether to refine landmarks around lips and eyes.
   * Adds ~80 additional landmarks for these regions.
   * Default: false
   */
  refineLandmarks?: boolean;

  /**
   * Minimum detection confidence (0-1).
   * Default: 0.5
   */
  minDetectionConfidence?: number;

  /**
   * Minimum tracking confidence (0-1).
   * Default: 0.5
   */
  minTrackingConfidence?: number;

  /**
   * If true, falls back to 68-point landmarks when FaceMesh fails.
   * Default: true
   */
  enableFallback?: boolean;
}

export class FaceMeshOptions {
  public readonly maxFaces: number;

  public readonly refineLandmarks: boolean;

  public readonly minDetectionConfidence: number;

  public readonly minTrackingConfidence: number;

  public readonly enableFallback: boolean;

  constructor(options: IFaceMeshOptions = {}) {
    this.maxFaces = options.maxFaces ?? 1;
    this.refineLandmarks = options.refineLandmarks ?? false;
    this.minDetectionConfidence = options.minDetectionConfidence ?? 0.5;
    this.minTrackingConfidence = options.minTrackingConfidence ?? 0.5;
    this.enableFallback = options.enableFallback ?? true;
  }
}

/**
 * Number of landmarks in different configurations.
 */
export const FACEMESH_LANDMARK_COUNTS = {
  /**
   * Base FaceMesh landmarks.
   */
  BASE: 468,

  /**
   * With refinement around lips and eyes.
   */
  REFINED: 478,

  /**
   * Legacy 68-point landmarks.
   */
  LEGACY_68: 68,

  /**
   * Legacy 5-point landmarks.
   */
  LEGACY_5: 5,
} as const;

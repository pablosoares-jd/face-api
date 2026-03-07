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
export declare class FaceMeshOptions {
    readonly maxFaces: number;
    readonly refineLandmarks: boolean;
    readonly minDetectionConfidence: number;
    readonly minTrackingConfidence: number;
    readonly enableFallback: boolean;
    constructor(options?: IFaceMeshOptions);
}
/**
 * Number of landmarks in different configurations.
 */
export declare const FACEMESH_LANDMARK_COUNTS: {
    /**
     * Base FaceMesh landmarks.
     */
    readonly BASE: 468;
    /**
     * With refinement around lips and eyes.
     */
    readonly REFINED: 478;
    /**
     * Legacy 68-point landmarks.
     */
    readonly LEGACY_68: 68;
    /**
     * Legacy 5-point landmarks.
     */
    readonly LEGACY_5: 5;
};

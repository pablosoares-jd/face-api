import { IDimensions, Point } from '../classes/index';
import { FaceLandmarks } from '../classes/FaceLandmarks';
import { FaceLandmarks68 } from '../classes/FaceLandmarks68';
/**
 * FaceMesh 468/478 point landmarks with 3D coordinates.
 */
export declare class FaceMeshLandmarks extends FaceLandmarks {
    private _zValues;
    constructor(positions: Point[], imageDims: IDimensions, zValues: number[], shift?: Point);
    /**
     * Get Z-coordinates for all landmarks.
     */
    get zValues(): number[];
    /**
     * Get 3D position for a specific landmark.
     */
    getPosition3D(idx: number): {
        x: number;
        y: number;
        z: number;
    };
    /**
     * Get all 3D positions.
     */
    get positions3D(): Array<{
        x: number;
        y: number;
        z: number;
    }>;
    /**
     * Helper to filter undefined values from Point arrays.
     */
    private filterPoints;
    /**
     * Get face oval landmarks (36 points).
     */
    getFaceOval(): Point[];
    /**
     * Get left eye landmarks.
     */
    getLeftEye(): Point[];
    /**
     * Get right eye landmarks.
     */
    getRightEye(): Point[];
    /**
     * Get left eyebrow landmarks.
     */
    getLeftEyebrow(): Point[];
    /**
     * Get right eyebrow landmarks.
     */
    getRightEyebrow(): Point[];
    /**
     * Get nose landmarks.
     */
    getNose(): Point[];
    /**
     * Get lips landmarks (inner + outer).
     */
    getLips(): Point[];
    /**
     * Get left iris landmarks (if refined).
     */
    getLeftIris(): Point[];
    /**
     * Get right iris landmarks (if refined).
     */
    getRightIris(): Point[];
    /**
     * Convert to 68-point FaceLandmarks68 for compatibility.
     */
    toLandmarks68(): FaceLandmarks68;
    /**
     * Estimate head pose from landmarks.
     * Returns rotation angles in degrees.
     */
    estimateHeadPose(): {
        pitch: number;
        yaw: number;
        roll: number;
    };
}

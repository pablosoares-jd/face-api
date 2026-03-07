import type { IDimensions } from '../classes/index';
import { Point } from '../classes/index';
import { FaceLandmarks } from '../classes/FaceLandmarks';
import { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import { FACEMESH_LANDMARK_COUNTS } from './FaceMeshOptions';
import { FACEMESH_TO_68_MAPPING } from './landmarkMapping';

/**
 * FaceMesh 468/478 point landmarks with 3D coordinates.
 */
export class FaceMeshLandmarks extends FaceLandmarks {
  private _zValues: number[];

  constructor(
    positions: Point[],
    imageDims: IDimensions,
    zValues: number[],
    shift: Point = new Point(0, 0),
  ) {
    super(positions, imageDims, shift);
    this._zValues = zValues;
  }

  /**
   * Get Z-coordinates for all landmarks.
   */
  public get zValues(): number[] {
    return this._zValues;
  }

  /**
   * Get 3D position for a specific landmark.
   */
  public getPosition3D(idx: number): { x: number; y: number; z: number } {
    const pos = this.positions[idx];
    if (!pos) {
      return { x: 0, y: 0, z: 0 };
    }
    return {
      x: pos.x,
      y: pos.y,
      z: this._zValues[idx] ?? 0,
    };
  }

  /**
   * Get all 3D positions.
   */
  public get positions3D(): Array<{ x: number; y: number; z: number }> {
    return this.positions.map((pos, idx) => ({
      x: pos.x,
      y: pos.y,
      z: this._zValues[idx] ?? 0,
    }));
  }

  /**
   * Helper to filter undefined values from Point arrays.
   */
  private filterPoints(indices: number[]): Point[] {
    return indices
      .map((i) => this.positions[i])
      .filter((p): p is Point => p !== undefined);
  }

  /**
   * Get face oval landmarks (36 points).
   */
  public getFaceOval(): Point[] {
    // MediaPipe face oval indices
    const faceOvalIndices = [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ];
    return this.filterPoints(faceOvalIndices);
  }

  /**
   * Get left eye landmarks.
   */
  public getLeftEye(): Point[] {
    const leftEyeIndices = [
      33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    ];
    return this.filterPoints(leftEyeIndices);
  }

  /**
   * Get right eye landmarks.
   */
  public getRightEye(): Point[] {
    const rightEyeIndices = [
      362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    ];
    return this.filterPoints(rightEyeIndices);
  }

  /**
   * Get left eyebrow landmarks.
   */
  public getLeftEyebrow(): Point[] {
    const leftEyebrowIndices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
    return this.filterPoints(leftEyebrowIndices);
  }

  /**
   * Get right eyebrow landmarks.
   */
  public getRightEyebrow(): Point[] {
    const rightEyebrowIndices = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
    return this.filterPoints(rightEyebrowIndices);
  }

  /**
   * Get nose landmarks.
   */
  public getNose(): Point[] {
    const noseIndices = [
      1, 2, 98, 327, 4, 5, 6, 168, 195, 197, 419, 351, 412, 343,
      437, 420, 456, 248, 281, 275, 274, 354, 370, 94, 19,
    ];
    return this.filterPoints(noseIndices);
  }

  /**
   * Get lips landmarks (inner + outer).
   */
  public getLips(): Point[] {
    const lipsIndices = [
      // Outer lips
      61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
      // Inner lips
      78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    ];
    return this.filterPoints(lipsIndices);
  }

  /**
   * Get left iris landmarks (if refined).
   */
  public getLeftIris(): Point[] {
    // Indices 468-472 are left iris (only with refinement)
    if (this.positions.length <= FACEMESH_LANDMARK_COUNTS.BASE) {
      return [];
    }
    return this.filterPoints([468, 469, 470, 471, 472]);
  }

  /**
   * Get right iris landmarks (if refined).
   */
  public getRightIris(): Point[] {
    // Indices 473-477 are right iris (only with refinement)
    if (this.positions.length <= FACEMESH_LANDMARK_COUNTS.BASE) {
      return [];
    }
    return this.filterPoints([473, 474, 475, 476, 477]);
  }

  /**
   * Convert to 68-point FaceLandmarks68 for compatibility.
   */
  public toLandmarks68(): FaceLandmarks68 {
    const points68 = FACEMESH_TO_68_MAPPING.map((meshIdx) => {
      const pos = this.positions[meshIdx];
      return pos || new Point(0, 0);
    });

    return new FaceLandmarks68(
      points68,
      { height: this.imageHeight, width: this.imageWidth },
    );
  }

  /**
   * Estimate head pose from landmarks.
   * Returns rotation angles in degrees.
   */
  public estimateHeadPose(): { pitch: number; yaw: number; roll: number } {
    // Use key landmarks to estimate pose
    const noseTip = this.positions[1];
    const leftEye = this.positions[33];
    const rightEye = this.positions[263];
    const chin = this.positions[152];

    if (!noseTip || !leftEye || !rightEye || !chin) {
      return { pitch: 0, yaw: 0, roll: 0 };
    }

    // Calculate eye center
    const eyeCenter = new Point(
      (leftEye.x + rightEye.x) / 2,
      (leftEye.y + rightEye.y) / 2,
    );

    // Estimate yaw (left-right rotation)
    const eyeWidth = Math.abs(rightEye.x - leftEye.x);
    const noseOffset = noseTip.x - eyeCenter.x;
    const yaw = Math.atan2(noseOffset, eyeWidth) * (180 / Math.PI);

    // Estimate pitch (up-down rotation)
    const faceHeight = Math.abs(chin.y - eyeCenter.y);
    const noseVertOffset = noseTip.y - eyeCenter.y;
    const pitch = Math.atan2(noseVertOffset - faceHeight * 0.3, faceHeight) * (180 / Math.PI);

    // Estimate roll (tilt)
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * (180 / Math.PI);

    return { pitch, yaw, roll };
  }
}

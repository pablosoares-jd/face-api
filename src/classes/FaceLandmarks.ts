import { minBbox } from '../ops/index';
import { getCenterPoint } from '../utils/index';
import type { IBoundingBox } from './BoundingBox';
import { Box } from './Box';
import type { IDimensions } from './Dimensions';
import { Dimensions } from './Dimensions';
import { FaceDetection } from './FaceDetection';
import { Point } from './Point';
import type { IRect } from './Rect';
import { Rect } from './Rect';

// face alignment constants
const relX = 0.5;
const relY = 0.43;
const relScale = 0.45;

export interface IFaceLandmarks {
  positions: Point[]
  shift: Point
}

export class FaceLandmarks implements IFaceLandmarks {
  protected _shift: Point;

  protected _positions: Point[];

  protected _imgDims: Dimensions;

  constructor(
    relativeFaceLandmarkPositions: Point[],
    imgDims: IDimensions,
    shift: Point = new Point(0, 0),
  ) {
    const { width, height } = imgDims;
    this._imgDims = new Dimensions(width, height);
    this._shift = shift;
    this._positions = relativeFaceLandmarkPositions.map(
      (pt) => pt.mul(new Point(width, height)).add(shift),
    );
  }

  public get shift(): Point { return new Point(this._shift.x, this._shift.y); }

  public get imageWidth(): number { return this._imgDims.width; }

  public get imageHeight(): number { return this._imgDims.height; }

  public get positions(): Point[] { return this._positions; }

  public get relativePositions(): Point[] {
    return this._positions.map(
      (pt) => pt.sub(this._shift).div(new Point(this.imageWidth, this.imageHeight)),
    );
  }

  public forSize<T extends FaceLandmarks>(width: number, height: number): T {
    return new (this.constructor as any)(
      this.relativePositions,
      { width, height },
    );
  }

  public shiftBy<T extends FaceLandmarks>(x: number, y: number): T {
    return new (this.constructor as any)(
      this.relativePositions,
      this._imgDims,
      new Point(x, y),
    );
  }

  public shiftByPoint<T extends FaceLandmarks>(pt: Point): T {
    return this.shiftBy(pt.x, pt.y);
  }

  /**
   * Aligns the face landmarks after face detection from the relative positions of the faces
   * bounding box, or it's current shift. This function should be used to align the face images
   * after face detection has been performed, before they are passed to the face recognition net.
   * This will make the computed face descriptor more accurate.
   *
   * @param detection (optional) The bounding box of the face or the face detection result. If
   * no argument was passed the position of the face landmarks are assumed to be relative to
   * it's current shift.
   * @returns The bounding box of the aligned face.
   */
  public align(
    detection?: FaceDetection | IRect | IBoundingBox | null,
    options: { useDlibAlignment?: boolean, minBoxPadding?: number } = { },
  ): Box {
    if (detection) {
      const box = detection instanceof FaceDetection
        ? detection.box.floor()
        : new Box(detection);

      return this.shiftBy(box.x, box.y).align(null, options);
    }

    const { useDlibAlignment, minBoxPadding } = { useDlibAlignment: false, minBoxPadding: 0.2, ...options };

    if (useDlibAlignment) {
      return this.alignDlib();
    }

    return this.alignMinBbox(minBoxPadding);
  }

  private alignDlib(): Box {
    const centers = this.getRefPointsForAlignment();

    // Validate reference points
    const [leftEyeCenter, rightEyeCenter, mouthCenter] = centers;
    if (!leftEyeCenter || !rightEyeCenter || !mouthCenter) {
      console.warn('FaceLandmarks.alignDlib: Missing reference points, falling back to minBbox');
      return this.alignMinBbox(0.2);
    }

    const distToMouth = (pt: Point) => mouthCenter.sub(pt).magnitude();
    const eyeToMouthDist = (distToMouth(leftEyeCenter) + distToMouth(rightEyeCenter)) / 2;

    // Validate eye-to-mouth distance
    if (!Number.isFinite(eyeToMouthDist) || eyeToMouthDist <= 0) {
      console.warn('FaceLandmarks.alignDlib: Invalid eye-to-mouth distance, falling back to minBbox');
      return this.alignMinBbox(0.2);
    }

    const size = Math.floor(eyeToMouthDist / relScale);

    // Ensure minimum size
    const validSize = Math.max(size, 1);

    const refPoint = getCenterPoint(centers);
    // TODO: pad in case rectangle is out of image bounds
    const x = Math.floor(Math.max(0, refPoint.x - (relX * validSize)));
    const y = Math.floor(Math.max(0, refPoint.y - (relY * validSize)));

    return new Rect(x, y, Math.min(validSize, this.imageWidth + x), Math.min(validSize, this.imageHeight + y));
  }

  private alignMinBbox(padding: number): Box {
    // Validate positions before computing bounding box
    if (!this.positions || this.positions.length === 0) {
      console.warn('FaceLandmarks.alignMinBbox: No landmark positions available');
      // Return a minimal valid box at origin
      return new Box({ x: 0, y: 0, width: 1, height: 1 });
    }

    const box = minBbox(this.positions);

    // Validate the resulting box has valid dimensions
    if (box.width <= 0 || box.height <= 0) {
      console.warn('FaceLandmarks.alignMinBbox: Invalid box dimensions from landmarks');
      return new Box({ x: box.x, y: box.y, width: 1, height: 1 });
    }

    return box.pad(box.width * padding, box.height * padding);
  }

  protected getRefPointsForAlignment(): Point[] {
    throw new Error('getRefPointsForAlignment not implemented by base class');
  }
}

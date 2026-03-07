import { Box } from './Box';
import type { IDimensions } from './Dimensions';
import { Dimensions } from './Dimensions';
import type { IRect, Rect } from './Rect';

export class ObjectDetection {
  private _score: number;

  private _classScore: number;

  private _className: string;

  private _box: Rect;

  private _imageDims: Dimensions;

  constructor(
    score: number,
    classScore: number,
    className: string,
    relativeBox: IRect,
    imageDims: IDimensions,
  ) {
    this._imageDims = new Dimensions(imageDims.width, imageDims.height);
    this._score = score;
    this._classScore = classScore;
    this._className = className;

    // Validate relativeBox coordinates before creating Box
    const validBox: IRect = {
      x: Number.isFinite(relativeBox.x) ? relativeBox.x : 0,
      y: Number.isFinite(relativeBox.y) ? relativeBox.y : 0,
      width: Number.isFinite(relativeBox.width) && relativeBox.width > 0 ? relativeBox.width : 1,
      height: Number.isFinite(relativeBox.height) && relativeBox.height > 0 ? relativeBox.height : 1,
    };

    this._box = new Box(validBox).rescale(this._imageDims);
  }

  public get score(): number { return this._score; }

  public get classScore(): number { return this._classScore; }

  public get className(): string { return this._className; }

  public get box(): Box { return this._box; }

  public get imageDims(): Dimensions { return this._imageDims; }

  public get imageWidth(): number { return this.imageDims.width; }

  public get imageHeight(): number { return this.imageDims.height; }

  public get relativeBox(): Box { return new Box(this._box).rescale(this.imageDims.reverse()); }

  public forSize(width: number, height: number): ObjectDetection {
    return new ObjectDetection(
      this.score,
      this.classScore,
      this.className,
      this.relativeBox,
      { width, height },
    );
  }
}

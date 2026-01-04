/**
 * Options for BlazeFace detector.
 */
export interface IBlazeFaceOptions {
  /**
   * Minimum confidence threshold for detections (0-1).
   * Default: 0.5
   */
  minConfidence?: number;

  /**
   * Maximum number of faces to detect.
   * Default: 10
   */
  maxResults?: number;

  /**
   * Input size for the model. BlazeFace supports 128 or 256.
   * Larger size = more accurate but slower.
   * Default: 128
   */
  inputSize?: 128 | 256;

  /**
   * IOU threshold for non-max suppression.
   * Default: 0.3
   */
  iouThreshold?: number;

  /**
   * If true, falls back to SSD MobileNetv1 when BlazeFace fails.
   * Default: true
   */
  enableFallback?: boolean;
}

export class BlazeFaceOptions {
  public readonly minConfidence: number;

  public readonly maxResults: number;

  public readonly inputSize: 128 | 256;

  public readonly iouThreshold: number;

  public readonly enableFallback: boolean;

  constructor(options: IBlazeFaceOptions = {}) {
    this.minConfidence = options.minConfidence ?? 0.5;
    this.maxResults = options.maxResults ?? 10;
    this.inputSize = options.inputSize ?? 128;
    this.iouThreshold = options.iouThreshold ?? 0.3;
    this.enableFallback = options.enableFallback ?? true;
  }
}

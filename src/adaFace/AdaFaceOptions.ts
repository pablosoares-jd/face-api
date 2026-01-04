/**
 * Options for AdaFace face recognition.
 */
export interface IAdaFaceOptions {
  /**
   * Input size for face crops. AdaFace uses 112x112.
   * Default: 112
   */
  inputSize?: number;

  /**
   * Descriptor size. AdaFace typically uses 512-d embeddings.
   * Default: 512
   */
  descriptorSize?: 128 | 256 | 512;

  /**
   * If true, falls back to FaceNet when AdaFace fails.
   * Default: true
   */
  enableFallback?: boolean;

  /**
   * Quality threshold for adaptive margin.
   * AdaFace adjusts margin based on image quality.
   * Default: 0.5
   */
  qualityThreshold?: number;

  /**
   * If true and both models are loaded, blend AdaFace and FaceNet descriptors.
   * This can improve accuracy for challenging images.
   * Default: false
   */
  blendDescriptors?: boolean;

  /**
   * Weight for AdaFace descriptor when blending (0-1).
   * FaceNet weight = 1 - blendWeight.
   * Default: 0.7 (AdaFace gets 70%, FaceNet 30%)
   */
  blendWeight?: number;
}

export class AdaFaceOptions {
  public readonly inputSize: number;

  public readonly descriptorSize: 128 | 256 | 512;

  public readonly enableFallback: boolean;

  public readonly qualityThreshold: number;

  public readonly blendDescriptors: boolean;

  public readonly blendWeight: number;

  constructor(options: IAdaFaceOptions = {}) {
    this.inputSize = options.inputSize ?? 112;
    this.descriptorSize = options.descriptorSize ?? 512;
    this.enableFallback = options.enableFallback ?? true;
    this.qualityThreshold = options.qualityThreshold ?? 0.5;
    this.blendDescriptors = options.blendDescriptors ?? false;
    this.blendWeight = Math.max(0, Math.min(1, options.blendWeight ?? 0.7));
  }
}

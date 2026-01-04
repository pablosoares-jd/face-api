import * as tf from '../../dist/tfjs.esm';

import { NetInput, TNetInput, toNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { FaceRecognitionNet } from '../faceRecognitionNet/FaceRecognitionNet';
import { AdaFaceOptions, IAdaFaceOptions } from './AdaFaceOptions';
import { extractParams } from './extractParams';
import { extractParamsFromWeightMap } from './extractParamsFromWeightMap';
import { NetParams } from './types';

/**
 * AdaFace - Adaptive Face Recognition for varying image quality.
 *
 * Performance: 99.82%+ on LFW (vs 99.63% for FaceNet)
 *
 * Key advantages:
 * - Quality-adaptive margin for better performance on low-quality images
 * - More robust to pose, lighting, and expression variations
 * - 512-dimensional embeddings for higher discriminative power
 * - Automatic fallback to FaceNet for compatibility
 *
 * @example
 * ```typescript
 * const recognizer = new AdaFace();
 * await recognizer.load('/models');
 *
 * const descriptor = await recognizer.computeFaceDescriptor(faceImage);
 * const distance = faceapi.euclideanDistance(descriptor1, descriptor2);
 * ```
 */
export class AdaFace extends NeuralNetwork<NetParams> {
  private _fallbackNet: FaceRecognitionNet | null = null;

  constructor() {
    super('AdaFace');
  }

  /**
   * Get the fallback recognizer (FaceNet/FaceRecognitionNet).
   */
  public get fallbackNet(): FaceRecognitionNet | null {
    return this._fallbackNet;
  }

  /**
   * Load AdaFace model with optional fallback to FaceNet.
   */
  public override async load(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    try {
      await super.load(weightsOrUrl);
    } catch {
      console.warn('AdaFace model not found, loading FaceNet fallback...');
      this._fallbackNet = new FaceRecognitionNet();
      await this._fallbackNet.load(weightsOrUrl);
    }
  }

  /**
   * Load fallback recognizer explicitly.
   */
  public async loadFallback(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    if (!this._fallbackNet) {
      this._fallbackNet = new FaceRecognitionNet();
    }
    await this._fallbackNet.load(weightsOrUrl);
  }

  /**
   * Check if the primary model is loaded.
   */
  public get isPrimaryLoaded(): boolean {
    return !!this.params;
  }

  /**
   * Check if any model is available for inference.
   */
  public override get isLoaded(): boolean {
    return this.isPrimaryLoaded || (this._fallbackNet?.isLoaded ?? false);
  }

  /**
   * Forward pass through the network.
   */
  public forwardInput(input: NetInput): tf.Tensor2D {
    const { params } = this;
    if (!params) {
      throw new Error('AdaFace - load model before inference');
    }

    return tf.tidy(() => {
      const batchTensor = tf.cast(input.toBatchTensor(112, true), 'float32');

      // AdaFace normalization: (x - 127.5) / 128
      const normalized = tf.div(tf.sub(batchTensor, 127.5), 128) as tf.Tensor4D;

      // Stem
      let x = this.convBnRelu(normalized, params.stem.conv1);
      x = this.convBnRelu(x, params.stem.conv2);
      x = this.convBnRelu(x, params.stem.conv3);

      // Backbone stages (IR-SE blocks)
      x = this.irSeBlock(x, params.stage1);
      x = this.irSeBlock(x, params.stage2);
      x = this.irSeBlock(x, params.stage3);
      x = this.irSeBlock(x, params.stage4);

      // Global pooling and FC
      const pooled = x.mean([1, 2]) as tf.Tensor2D;
      const embedding = tf.matMul(pooled, params.fc.weights);
      const withBias = tf.add(embedding, params.fc.bias);

      // L2 normalize using sqrt(sum(x^2))
      const squared = tf.mul(withBias, withBias);
      const sumSquared = squared.sum(1, true);
      const l2Norm = tf.sqrt(sumSquared);
      return tf.div(withBias, l2Norm) as tf.Tensor2D;
    });
  }

  /**
   * Convolution + BatchNorm + ReLU block.
   */
  private convBnRelu(x: tf.Tensor4D, convParams: NetParams['stem']['conv1']): tf.Tensor4D {
    return tf.tidy(() => {
      let out = tf.conv2d(x, convParams.weights, convParams.stride || 1, 'same');
      out = tf.batchNorm(
        out,
        convParams.bn_mean,
        convParams.bn_variance,
        convParams.bn_offset,
        convParams.bn_scale,
        0.001,
      );
      return tf.relu(out) as tf.Tensor4D;
    });
  }

  /**
   * IR-SE (Inverted Residual with Squeeze-Excitation) block.
   */
  private irSeBlock(x: tf.Tensor4D, blockParams: NetParams['stage1']): tf.Tensor4D {
    return tf.tidy(() => {
      // Depthwise separable convolution
      let out = tf.depthwiseConv2d(x, blockParams.depthwise, 1, 'same');
      out = tf.batchNorm(
        out,
        blockParams.bn1_mean,
        blockParams.bn1_variance,
        blockParams.bn1_offset,
        blockParams.bn1_scale,
        0.001,
      );
      out = tf.relu(out);

      // Squeeze-Excitation
      const se = this.squeezeExcitation(out, blockParams.se);
      out = tf.mul(out, se) as tf.Tensor4D;

      // Pointwise
      out = tf.conv2d(out, blockParams.pointwise, 1, 'same');
      out = tf.batchNorm(
        out,
        blockParams.bn2_mean,
        blockParams.bn2_variance,
        blockParams.bn2_offset,
        blockParams.bn2_scale,
        0.001,
      );

      // Residual connection
      return tf.add(x, out) as tf.Tensor4D;
    });
  }

  /**
   * Squeeze-Excitation module.
   */
  private squeezeExcitation(x: tf.Tensor4D, seParams: NetParams['stage1']['se']): tf.Tensor4D {
    return tf.tidy(() => {
      // Global average pooling
      const squeezed = x.mean([1, 2], true);
      const batchSize = squeezed.shape[0] ?? 1;
      const channels = squeezed.shape[3] ?? 1;

      // FC -> ReLU -> FC -> Sigmoid
      let se = tf.matMul(squeezed.reshape([batchSize, channels]), seParams.fc1);
      se = tf.relu(se);
      se = tf.matMul(se, seParams.fc2);
      se = tf.sigmoid(se);

      // Reshape for broadcasting
      const seShape1 = se.shape[1] ?? 1;
      return se.reshape([batchSize, 1, 1, seShape1]) as tf.Tensor4D;
    });
  }

  /**
   * Compute face descriptor (embedding) for recognition.
   */
  public async computeFaceDescriptor(
    input: TNetInput,
    options: IAdaFaceOptions = {},
  ): Promise<Float32Array | Float32Array[]> {
    const opts = new AdaFaceOptions(options);

    // Use fallback if primary not loaded
    if (!this.isPrimaryLoaded && this._fallbackNet?.isLoaded) {
      const result = await this._fallbackNet.computeFaceDescriptor(input);
      return result;
    }

    if (!this.isPrimaryLoaded) {
      throw new Error('AdaFace - no model loaded. Call load() first.');
    }

    const netInput = await toNetInput(input);
    const descriptorTensors = tf.tidy(() => tf.unstack(this.forwardInput(netInput)));

    try {
      const descriptors = await Promise.all(
        descriptorTensors.map((t: tf.Tensor) => t.data()),
      ) as Float32Array[];

      // If fallback enabled and we have it loaded, could blend results
      if (opts.enableFallback && this._fallbackNet?.isLoaded) {
        // Future: implement quality-based blending
      }

      return netInput.isBatchInput ? descriptors : descriptors[0];
    } finally {
      descriptorTensors.forEach((t: tf.Tensor) => t.dispose());
    }
  }

  public async forward(input: TNetInput): Promise<tf.Tensor2D> {
    return this.forwardInput(await toNetInput(input));
  }

  /**
   * Dispose of resources.
   */
  public override dispose(throwOnRedispose = true): void {
    if (this.isPrimaryLoaded) {
      super.dispose(throwOnRedispose);
    }
    if (this._fallbackNet) {
      this._fallbackNet.dispose(throwOnRedispose);
      this._fallbackNet = null;
    }
  }

  protected getDefaultModelName(): string {
    return 'adaface_model';
  }

  protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeightMap(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights);
  }
}

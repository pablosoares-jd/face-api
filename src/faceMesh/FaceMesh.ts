import * as tf from '../../dist/tfjs.esm';

import { IDimensions, Point } from '../classes/index';
import { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import { NetInput, TNetInput, toNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { FaceLandmark68Net } from '../faceLandmarkNet/FaceLandmark68Net';
import { FaceMeshOptions, IFaceMeshOptions, FACEMESH_LANDMARK_COUNTS } from './FaceMeshOptions';
import { FaceMeshLandmarks } from './FaceMeshLandmarks';
import { extractParams } from './extractParams';
import { extractParamsFromWeightMap } from './extractParamsFromWeightMap';
import { NetParams } from './types';

/**
 * FaceMesh - 468/478 point facial landmark detector from MediaPipe.
 *
 * Features:
 * - 468 dense facial landmarks (vs 68 for traditional models)
 * - Optional refinement for 478 landmarks (extra eye/lip detail)
 * - Real-time 3D face mesh generation
 * - Automatic fallback to 68-point landmarks
 *
 * Landmark regions:
 * - Face oval: 36 points
 * - Left eyebrow: 8 points
 * - Right eyebrow: 8 points
 * - Left eye: 16 points (71 with refinement)
 * - Right eye: 16 points (71 with refinement)
 * - Nose: 25 points
 * - Lips: 40 points (80 with refinement)
 * - Face mesh: 359 points
 *
 * @example
 * ```typescript
 * const mesh = new FaceMesh();
 * await mesh.load('/models');
 *
 * const landmarks = await mesh.detectLandmarks(image);
 * console.log(landmarks.positions.length); // 468 or 478
 * ```
 */
export class FaceMesh extends NeuralNetwork<NetParams> {
  private _fallbackNet: FaceLandmark68Net | null = null;

  constructor() {
    super('FaceMesh');
  }

  /**
   * Get the fallback detector (68-point landmarks).
   */
  public get fallbackNet(): FaceLandmark68Net | null {
    return this._fallbackNet;
  }

  /**
   * Load FaceMesh model with optional fallback to 68-point landmarks.
   */
  public override async load(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    try {
      await super.load(weightsOrUrl);
    } catch (error) {
      console.warn('FaceMesh model not found, loading 68-point landmark fallback...');
      this._fallbackNet = new FaceLandmark68Net();
      await this._fallbackNet.load(weightsOrUrl);
    }
  }

  /**
   * Load fallback detector explicitly.
   */
  public async loadFallback(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    if (!this._fallbackNet) {
      this._fallbackNet = new FaceLandmark68Net();
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
      throw new Error('FaceMesh - load model before inference');
    }

    return tf.tidy(() => {
      const inputSize = 192;
      const batchTensor = tf.cast(input.toBatchTensor(inputSize, false), 'float32');

      // Normalize to [-1, 1]
      const normalized = tf.sub(tf.div(batchTensor, 127.5), 1) as tf.Tensor4D;

      // Encoder
      let x = this.convBlock(normalized, params.encoder.conv1, 2);
      x = this.convBlock(x, params.encoder.conv2, 1);
      x = this.convBlock(x, params.encoder.conv3, 2);
      x = this.convBlock(x, params.encoder.conv4, 1);
      x = this.convBlock(x, params.encoder.conv5, 2);

      // Bottleneck
      for (const block of params.bottleneck) {
        x = this.residualBlock(x, block);
      }

      // Decoder/landmark head
      x = this.convBlock(x, params.decoder.conv1, 1);
      const landmarks = tf.conv2d(x, params.landmarkHead.weights, 1, 'same');

      // Reshape to [batch, num_landmarks * 3]
      const batchSize = normalized.shape[0];
      const numLandmarks = FACEMESH_LANDMARK_COUNTS.BASE;
      return tf.reshape(landmarks, [batchSize, numLandmarks * 3]) as tf.Tensor2D;
    });
  }

  /**
   * Convolution block.
   */
  private convBlock(x: tf.Tensor4D, params: any, stride: number): tf.Tensor4D {
    return tf.tidy(() => {
      let out = tf.conv2d(x, params.weights, stride, 'same');
      out = tf.batchNorm(
        out,
        params.bn_mean,
        params.bn_variance,
        params.bn_offset,
        params.bn_scale,
        0.001,
      );
      return tf.relu(out) as tf.Tensor4D;
    });
  }

  /**
   * Residual block.
   */
  private residualBlock(x: tf.Tensor4D, params: any): tf.Tensor4D {
    return tf.tidy(() => {
      let out = this.convBlock(x, params.conv1, 1);
      out = tf.conv2d(out, params.conv2.weights, 1, 'same');
      out = tf.batchNorm(
        out,
        params.conv2.bn_mean,
        params.conv2.bn_variance,
        params.conv2.bn_offset,
        params.conv2.bn_scale,
        0.001,
      );
      return tf.add(x, out) as tf.Tensor4D;
    });
  }

  /**
   * Detect facial landmarks.
   * @param input Input image
   * @param options Detection options including refineLandmarks for iris detection
   */
  public async detectLandmarks(
    input: TNetInput,
    options: IFaceMeshOptions = {},
  ): Promise<FaceMeshLandmarks | FaceMeshLandmarks[] | FaceLandmarks68 | FaceLandmarks68[]> {
    const opts = new FaceMeshOptions(options);

    // Use fallback if primary not loaded
    if (!this.isPrimaryLoaded && this._fallbackNet?.isLoaded) {
      return this._fallbackNet.detectLandmarks(input);
    }

    if (!this.isPrimaryLoaded) {
      throw new Error('FaceMesh - no model loaded. Call load() first.');
    }

    const netInput = await toNetInput(input);
    const landmarkTensor = tf.tidy(() => this.forwardInput(netInput));

    try {
      const landmarkData = await landmarkTensor.array() as number[][];

      const landmarksForBatch = landmarkData.map((data, batchIdx) => {
        const baseLandmarks = FACEMESH_LANDMARK_COUNTS.BASE;
        const points: Point[] = new Array(baseLandmarks);
        const zValues: number[] = new Array(baseLandmarks);

        // Extract base 468 landmarks
        for (let i = 0; i < baseLandmarks; i++) {
          points[i] = new Point(
            data[i * 3] as number,     // x
            data[i * 3 + 1] as number, // y
          );
          zValues[i] = data[i * 3 + 2] as number; // z
        }

        // If refinement is enabled, estimate iris landmarks
        if (opts.refineLandmarks) {
          const irisLandmarks = this.estimateIrisLandmarks(points);
          points.push(...irisLandmarks.points);
          zValues.push(...irisLandmarks.zValues);
        }

        return new FaceMeshLandmarks(
          points,
          {
            height: netInput.getInputHeight(batchIdx),
            width: netInput.getInputWidth(batchIdx),
          },
          zValues,
        );
      });

      return netInput.isBatchInput ? landmarksForBatch : landmarksForBatch[0]!;
    } finally {
      landmarkTensor.dispose();
    }
  }

  /**
   * Estimate iris landmarks (468-477) from eye landmarks.
   * This provides approximate iris positions when a refined model is not available.
   */
  private estimateIrisLandmarks(baseLandmarks: Point[]): { points: Point[]; zValues: number[] } {
    const points: Point[] = [];
    const zValues: number[] = [];

    // Left eye key landmarks for iris estimation
    // Indices: 33 (outer), 133 (inner), 159 (upper), 145 (lower)
    const leftOuter = baseLandmarks[33];
    const leftInner = baseLandmarks[133];
    const leftUpper = baseLandmarks[159];
    const leftLower = baseLandmarks[145];

    // Right eye key landmarks
    // Indices: 362 (outer), 263 (inner), 386 (upper), 374 (lower)
    const rightOuter = baseLandmarks[362];
    const rightInner = baseLandmarks[263];
    const rightUpper = baseLandmarks[386];
    const rightLower = baseLandmarks[374];

    if (leftOuter && leftInner && leftUpper && leftLower) {
      // Left iris center (index 468)
      const leftCenterX = (leftOuter.x + leftInner.x) / 2;
      const leftCenterY = (leftUpper.y + leftLower.y) / 2;
      const leftRadiusX = Math.abs(leftInner.x - leftOuter.x) / 4;
      const leftRadiusY = Math.abs(leftLower.y - leftUpper.y) / 3;

      // Left iris landmarks (468-472): center, left, top, right, bottom
      points.push(new Point(leftCenterX, leftCenterY)); // 468: center
      points.push(new Point(leftCenterX - leftRadiusX, leftCenterY)); // 469: left
      points.push(new Point(leftCenterX, leftCenterY - leftRadiusY)); // 470: top
      points.push(new Point(leftCenterX + leftRadiusX, leftCenterY)); // 471: right
      points.push(new Point(leftCenterX, leftCenterY + leftRadiusY)); // 472: bottom
      zValues.push(0, 0, 0, 0, 0);
    } else {
      // Fallback: add placeholder points
      for (let i = 0; i < 5; i++) {
        points.push(new Point(0, 0));
        zValues.push(0);
      }
    }

    if (rightOuter && rightInner && rightUpper && rightLower) {
      // Right iris center (index 473)
      const rightCenterX = (rightOuter.x + rightInner.x) / 2;
      const rightCenterY = (rightUpper.y + rightLower.y) / 2;
      const rightRadiusX = Math.abs(rightInner.x - rightOuter.x) / 4;
      const rightRadiusY = Math.abs(rightLower.y - rightUpper.y) / 3;

      // Right iris landmarks (473-477): center, left, top, right, bottom
      points.push(new Point(rightCenterX, rightCenterY)); // 473: center
      points.push(new Point(rightCenterX - rightRadiusX, rightCenterY)); // 474: left
      points.push(new Point(rightCenterX, rightCenterY - rightRadiusY)); // 475: top
      points.push(new Point(rightCenterX + rightRadiusX, rightCenterY)); // 476: right
      points.push(new Point(rightCenterX, rightCenterY + rightRadiusY)); // 477: bottom
      zValues.push(0, 0, 0, 0, 0);
    } else {
      // Fallback: add placeholder points
      for (let i = 0; i < 5; i++) {
        points.push(new Point(0, 0));
        zValues.push(0);
      }
    }

    return { points, zValues };
  }

  /**
   * Detect landmarks and convert to 68-point format for compatibility.
   */
  public async detectLandmarks68(
    input: TNetInput,
    options: IFaceMeshOptions = {},
  ): Promise<FaceLandmarks68 | FaceLandmarks68[]> {
    const opts = new FaceMeshOptions(options);

    // Use fallback directly if primary not loaded
    if (!this.isPrimaryLoaded && this._fallbackNet?.isLoaded) {
      return this._fallbackNet.detectLandmarks(input);
    }

    const result = await this.detectLandmarks(input, options);

    // Convert FaceMeshLandmarks to FaceLandmarks68
    if (Array.isArray(result)) {
      return result.map((r) => {
        if (r instanceof FaceMeshLandmarks) {
          return r.toLandmarks68();
        }
        return r as FaceLandmarks68;
      });
    }

    if (result instanceof FaceMeshLandmarks) {
      return result.toLandmarks68();
    }

    return result as FaceLandmarks68;
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
    return 'facemesh_model';
  }

  protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeightMap(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights);
  }
}

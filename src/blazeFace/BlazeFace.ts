import * as tf from '@tensorflow/tfjs';

import { Point, Rect } from '../classes/index';
import { FaceDetection } from '../classes/FaceDetection';
import type { NetInput, TNetInput } from '../dom/index';
import { toNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { nonMaxSuppressionFast } from '../ops/nonMaxSuppression';
import { SsdMobilenetv1 } from '../ssdMobilenetv1/SsdMobilenetv1';
import type { IBlazeFaceOptions } from './BlazeFaceOptions';
import { BlazeFaceOptions } from './BlazeFaceOptions';
import { extractParams } from './extractParams';
import { extractParamsFromWeightMap } from './extractParamsFromWeightMap';
import type { NetParams, ConvBlockParams } from './types';

/**
 * Anchor configuration for BlazeFace.
 */
interface Anchor {
  x: number;
  y: number;
}

/**
 * BlazeFace facial keypoints.
 * These are the 6 keypoints returned by BlazeFace.
 */
export interface BlazeFaceKeypoints {
  /** Right eye center */
  rightEye: Point;
  /** Left eye center */
  leftEye: Point;
  /** Nose tip */
  noseTip: Point;
  /** Mouth center */
  mouthCenter: Point;
  /** Right ear tragion */
  rightEar: Point;
  /** Left ear tragion */
  leftEar: Point;
}

/**
 * BlazeFace detection result with keypoints.
 */
export class BlazeFaceDetection extends FaceDetection {
  private _keypoints: BlazeFaceKeypoints;

  constructor(
    score: number,
    relativeBox: Rect,
    imageDims: { width: number; height: number },
    keypoints: BlazeFaceKeypoints,
  ) {
    super(score, relativeBox, imageDims);
    this._keypoints = keypoints;
  }

  /**
   * Get the 6 facial keypoints.
   */
  public get keypoints(): BlazeFaceKeypoints {
    return this._keypoints;
  }

  /**
   * Get keypoints as an array of Points.
   */
  public get keypointsArray(): Point[] {
    return [
      this._keypoints.rightEye,
      this._keypoints.leftEye,
      this._keypoints.noseTip,
      this._keypoints.mouthCenter,
      this._keypoints.rightEar,
      this._keypoints.leftEar,
    ];
  }
}

/**
 * BlazeFace - Ultra-fast face detector from MediaPipe.
 *
 * Performance: 200-1000+ FPS (vs 20-40 FPS for SSD MobileNetv1)
 * Accuracy: ~98% (vs ~91% for SSD MobileNetv1)
 *
 * Features:
 * - Optimized for real-time face detection
 * - Returns 6 facial keypoints (eyes, ears, nose, mouth)
 * - Automatic fallback to SSD MobileNetv1 if model not loaded
 *
 * @example
 * ```typescript
 * const detector = new BlazeFace();
 * await detector.load('/models');
 *
 * const faces = await detector.locateFaces(image, { minConfidence: 0.7 });
 * ```
 */
export class BlazeFace extends NeuralNetwork<NetParams> {
  private _fallbackNet: SsdMobilenetv1 | null = null;

  private _graphModel: tf.GraphModel | null = null;

  private _anchors: Anchor[] = [];

  private _currentInputSize: 128 | 256 = 128;

  private static readonly STRIDES = [8, 16] as const;

  private static readonly ANCHORS_PER_STRIDE = [2, 6] as const;

  constructor() {
    super('BlazeFace');
  }

  /**
   * Check if graph model is loaded (for graph-model format).
   */
  public get isGraphModelLoaded(): boolean {
    return this._graphModel !== null;
  }

  /**
   * Get the fallback detector (SSD MobileNetv1).
   */
  public get fallbackNet(): SsdMobilenetv1 | null {
    return this._fallbackNet;
  }

  /**
   * Generate anchors for the model.
   */
  private generateAnchors(inputSize: number): Anchor[] {
    const anchors: Anchor[] = [];

    for (let i = 0; i < BlazeFace.STRIDES.length; i++) {
      const stride = BlazeFace.STRIDES[i]!;
      const gridSize = Math.ceil(inputSize / stride);
      const numAnchors = BlazeFace.ANCHORS_PER_STRIDE[i]!;

      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          for (let a = 0; a < numAnchors; a++) {
            anchors.push({
              x: (x + 0.5) / gridSize,
              y: (y + 0.5) / gridSize,
            });
          }
        }
      }
    }

    return anchors;
  }

  /**
   * Load BlazeFace model with optional fallback to SSD MobileNetv1.
   * Tries graph-model format first, then layers-model, then falls back to SSD.
   */
  public override async load(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    // If it's a string URL, try loading as graph-model first
    if (typeof weightsOrUrl === 'string') {
      try {
        // Try graph-model format (blazeface_model-weights_manifest.json has format: "graph-model")
        const graphModelUrl = weightsOrUrl.endsWith('/')
          ? `${weightsOrUrl}blazeface_model-weights_manifest.json`
          : `${weightsOrUrl}/blazeface_model-weights_manifest.json`;

        this._graphModel = await tf.loadGraphModel(graphModelUrl);
        this._anchors = this.generateAnchors(128);
        console.info('BlazeFace loaded as graph-model');
        return;
      } catch (graphErr) {
        // Graph model failed, try layers-model
        console.info('BlazeFace graph-model not found, trying layers-model...', graphErr);
      }
    }

    // Try layers-model format
    try {
      await super.load(weightsOrUrl);
      this._anchors = this.generateAnchors(128);
      console.info('BlazeFace loaded as layers-model');
    } catch {
      console.warn('BlazeFace model not found, loading SSD MobileNetv1 fallback...');
      this._fallbackNet = new SsdMobilenetv1();
      await this._fallbackNet.load(weightsOrUrl);
    }
  }

  /**
   * Load fallback detector explicitly.
   */
  public async loadFallback(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    if (!this._fallbackNet) {
      this._fallbackNet = new SsdMobilenetv1();
    }
    await this._fallbackNet.load(weightsOrUrl);
  }

  /**
   * Check if the primary model is loaded (either graph-model or layers-model).
   */
  public get isPrimaryLoaded(): boolean {
    return this._graphModel !== null || !!this.params;
  }

  /**
   * Check if any model is available for inference.
   */
  public override get isLoaded(): boolean {
    return this.isPrimaryLoaded || (this._fallbackNet?.isLoaded ?? false);
  }

  /**
   * Check which model type is loaded.
   */
  public get modelType(): 'graph' | 'layers' | 'fallback' | 'none' {
    if (this._graphModel) return 'graph';
    if (this.params) return 'layers';
    if (this._fallbackNet?.isLoaded) return 'fallback';
    return 'none';
  }

  /**
   * Forward pass through the graph model.
   * Graph model output format: [batch, num_detections, 17]
   * - 4 values for bounding box (center_x, center_y, width, height)
   * - 1 value for score
   * - 12 values for keypoints (6 points × 2 coords)
   */
  private forwardGraphModel(input: NetInput, inputSize: 128 | 256 = 128): { rawBoxes: tf.Tensor2D; scores: tf.Tensor1D } {
    if (!this._graphModel) {
      throw new Error('BlazeFace graph model not loaded');
    }

    return tf.tidy(() => {
      const batchTensor = tf.cast(input.toBatchTensor(inputSize, false), 'float32');

      // Normalize to [0, 1] - same as MediaPipe BlazeFace
      const normalized = tf.div(batchTensor, 255) as tf.Tensor4D;

      // Run inference
      const output = this._graphModel!.predict(normalized) as tf.Tensor;

      // Output shape is [1, num_anchors, 17]
      // Slice into boxes (first 16 values) and scores (last value)
      const outputReshaped = tf.reshape(output, [-1, 17]);

      // Extract boxes (indices 0-15) and scores (index 16, but sometimes in different position)
      // BlazeFace outputs: [x_center, y_center, w, h, kp1_x, kp1_y, ..., kp6_x, kp6_y, score]
      // We need to reshape to match our expected format
      const rawBoxes = tf.slice(outputReshaped, [0, 0], [-1, 16]) as tf.Tensor2D;
      const scoresRaw = tf.slice(outputReshaped, [0, 16], [-1, 1]);
      const scores = tf.sigmoid(scoresRaw).squeeze([1]) as tf.Tensor1D;

      return { rawBoxes, scores };
    });
  }

  /**
   * Forward pass through the network.
   * @param input The input tensor
   * @param inputSize The input size (128 or 256)
   * @returns Raw regressor output (16 values per detection) and scores
   */
  public forwardInput(input: NetInput, inputSize: 128 | 256 = 128): { rawBoxes: tf.Tensor2D; scores: tf.Tensor1D } {
    // Use graph model if available
    if (this._graphModel) {
      // Regenerate anchors if input size changed
      if (this._anchors.length === 0 || this._currentInputSize !== inputSize) {
        this._anchors = this.generateAnchors(inputSize);
        this._currentInputSize = inputSize;
      }
      return this.forwardGraphModel(input, inputSize);
    }

    // Use layers model
    const { params } = this;
    if (!params) {
      throw new Error('BlazeFace - load model before inference');
    }

    // Regenerate anchors if input size changed
    if (this._anchors.length === 0 || this._currentInputSize !== inputSize) {
      this._anchors = this.generateAnchors(inputSize);
      this._currentInputSize = inputSize;
    }

    return tf.tidy(() => {
      const batchTensor = tf.cast(input.toBatchTensor(inputSize, false), 'float32');

      // Normalize to [0, 1]
      const normalized = tf.div(batchTensor, 255) as tf.Tensor4D;

      // Backbone convolutions
      let x = this.convBlock(normalized, params.conv1, 2);
      x = this.convBlock(x, params.conv2, 1);
      x = this.convBlock(x, params.conv3, 2);
      x = this.convBlock(x, params.conv4, 1);
      x = this.convBlock(x, params.conv5, 2);
      x = this.convBlock(x, params.conv6, 1);
      x = this.convBlock(x, params.conv7, 1);
      x = this.convBlock(x, params.conv8, 1);

      // Detection heads
      const classifier = tf.conv2d(x, params.classifierHead.weights, 1, 'same');
      const regressor = tf.conv2d(x, params.regressorHead.weights, 1, 'same');

      // Reshape outputs
      const classifierFlat = tf.reshape(classifier, [-1, 1]);
      const regressorFlat = tf.reshape(regressor, [-1, 16]) as tf.Tensor2D;

      // Apply sigmoid to scores
      const scores = tf.sigmoid(classifierFlat).squeeze([1]) as tf.Tensor1D;

      return { rawBoxes: regressorFlat, scores };
    });
  }

  /**
   * Convolution block with batch norm and activation.
   */
  private convBlock(x: tf.Tensor4D, blockParams: ConvBlockParams, stride: number): tf.Tensor4D {
    return tf.tidy(() => {
      let out = tf.depthwiseConv2d(x, blockParams.depthwise, stride, 'same');
      out = tf.add(out, blockParams.depthwiseBias);
      out = tf.relu(out);

      out = tf.conv2d(out, blockParams.pointwise, 1, 'same');
      out = tf.add(out, blockParams.pointwiseBias);
      return tf.relu(out) as tf.Tensor4D;
    });
  }

  /**
   * Decoded detection result with box and keypoints.
   */
  private async decodeDetections(
    rawBoxes: tf.Tensor2D,
    inputSize: number,
  ): Promise<{ boxes: number[][]; keypoints: BlazeFaceKeypoints[] }> {
    const boxesData = await rawBoxes.array() as number[][];
    const decodedBoxes: number[][] = [];
    const decodedKeypoints: BlazeFaceKeypoints[] = [];

    const numBoxes = Math.min(this._anchors.length, boxesData.length);
    for (let i = 0; i < numBoxes; i++) {
      const anchor = this._anchors[i];
      const box = boxesData[i];
      if (!anchor || !box) continue;

      // Decode bounding box
      const cx = anchor.x + (box[0] ?? 0) / inputSize;
      const cy = anchor.y + (box[1] ?? 0) / inputSize;
      const w = (box[2] ?? 0) / inputSize;
      const h = (box[3] ?? 0) / inputSize;

      decodedBoxes.push([
        cy - h / 2, // top
        cx - w / 2, // left
        cy + h / 2, // bottom
        cx + w / 2, // right
      ]);

      // Decode 6 keypoints (each has x, y offset from anchor)
      // Keypoint order: right eye, left eye, nose, mouth, right ear, left ear
      const keypoints: BlazeFaceKeypoints = {
        rightEye: new Point(
          anchor.x + (box[4] ?? 0) / inputSize,
          anchor.y + (box[5] ?? 0) / inputSize,
        ),
        leftEye: new Point(
          anchor.x + (box[6] ?? 0) / inputSize,
          anchor.y + (box[7] ?? 0) / inputSize,
        ),
        noseTip: new Point(
          anchor.x + (box[8] ?? 0) / inputSize,
          anchor.y + (box[9] ?? 0) / inputSize,
        ),
        mouthCenter: new Point(
          anchor.x + (box[10] ?? 0) / inputSize,
          anchor.y + (box[11] ?? 0) / inputSize,
        ),
        rightEar: new Point(
          anchor.x + (box[12] ?? 0) / inputSize,
          anchor.y + (box[13] ?? 0) / inputSize,
        ),
        leftEar: new Point(
          anchor.x + (box[14] ?? 0) / inputSize,
          anchor.y + (box[15] ?? 0) / inputSize,
        ),
      };
      decodedKeypoints.push(keypoints);
    }

    return {
      boxes: decodedBoxes.length > 0 ? decodedBoxes : [[0, 0, 0, 0]],
      keypoints: decodedKeypoints,
    };
  }

  /**
   * Detect faces in an image.
   * Returns BlazeFaceDetection objects with 6 facial keypoints.
   */
  public async locateFaces(input: TNetInput, options: IBlazeFaceOptions = {}): Promise<BlazeFaceDetection[]> {
    const opts = new BlazeFaceOptions(options);

    // Use fallback if primary not loaded (returns FaceDetection without keypoints)
    if (!this.isPrimaryLoaded && this._fallbackNet?.isLoaded) {
      const fallbackResults = await this._fallbackNet.locateFaces(input, {
        minConfidence: opts.minConfidence,
        maxResults: opts.maxResults,
      });
      // Convert to BlazeFaceDetection with default keypoints
      return fallbackResults.map((det) => new BlazeFaceDetection(
        det.score,
        det.relativeBox,
        { width: det.imageWidth, height: det.imageHeight },
        this.createDefaultKeypoints(det.relativeBox),
      ));
    }

    if (!this.isPrimaryLoaded) {
      throw new Error('BlazeFace - no model loaded. Call load() first.');
    }

    const netInput = await toNetInput(input);
    const { rawBoxes, scores } = this.forwardInput(netInput, opts.inputSize);

    try {
      const scoresData = await scores.data();

      // Defensive null check for scoresData
      if (!scoresData || scoresData.length === 0) {
        console.warn('BlazeFace: No scores data returned from model');
        return [];
      }

      // Decode detections with keypoints
      const { boxes: decodedBoxes, keypoints: decodedKeypoints } = await this.decodeDetections(rawBoxes, opts.inputSize);

      // Non-max suppression using unified implementation
      const selectedIndices = nonMaxSuppressionFast(
        decodedBoxes,
        Array.from(scoresData),
        opts.maxResults,
        opts.iouThreshold,
        opts.minConfidence,
      );

      const reshapedDims = netInput.getReshapedInputDimensions(0);
      const padX = opts.inputSize / reshapedDims.width;
      const padY = opts.inputSize / reshapedDims.height;
      const imageDims = { height: netInput.getInputHeight(0), width: netInput.getInputWidth(0) };

      const results: BlazeFaceDetection[] = [];
      for (const idx of selectedIndices) {
        const boxData = decodedBoxes[idx];
        const kpData = decodedKeypoints[idx];
        if (!boxData || !kpData) continue;

        const top = Math.max(0, boxData[0] ?? 0) * padY;
        const left = Math.max(0, boxData[1] ?? 0) * padX;
        const bottom = Math.min(1.0, boxData[2] ?? 0) * padY;
        const right = Math.min(1.0, boxData[3] ?? 0) * padX;

        // Scale keypoints to image coordinates
        const scaledKeypoints: BlazeFaceKeypoints = {
          rightEye: new Point(kpData.rightEye.x * padX, kpData.rightEye.y * padY),
          leftEye: new Point(kpData.leftEye.x * padX, kpData.leftEye.y * padY),
          noseTip: new Point(kpData.noseTip.x * padX, kpData.noseTip.y * padY),
          mouthCenter: new Point(kpData.mouthCenter.x * padX, kpData.mouthCenter.y * padY),
          rightEar: new Point(kpData.rightEar.x * padX, kpData.rightEar.y * padY),
          leftEar: new Point(kpData.leftEar.x * padX, kpData.leftEar.y * padY),
        };

        results.push(new BlazeFaceDetection(
          scoresData[idx] ?? 0,
          new Rect(left, top, right - left, bottom - top),
          imageDims,
          scaledKeypoints,
        ));
      }

      // Try fallback if no results and fallback enabled
      if (results.length === 0 && opts.enableFallback && this._fallbackNet?.isLoaded) {
        const fallbackResults = await this._fallbackNet.locateFaces(input, {
          minConfidence: opts.minConfidence,
          maxResults: opts.maxResults,
        });
        return fallbackResults.map((det) => new BlazeFaceDetection(
          det.score,
          det.relativeBox,
          { width: det.imageWidth, height: det.imageHeight },
          this.createDefaultKeypoints(det.relativeBox),
        ));
      }

      return results;
    } finally {
      rawBoxes.dispose();
      scores.dispose();
    }
  }

  /**
   * Create default keypoints based on bounding box (for fallback).
   */
  private createDefaultKeypoints(box: Rect): BlazeFaceKeypoints {
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    return {
      rightEye: new Point(cx - box.width * 0.15, cy - box.height * 0.15),
      leftEye: new Point(cx + box.width * 0.15, cy - box.height * 0.15),
      noseTip: new Point(cx, cy),
      mouthCenter: new Point(cx, cy + box.height * 0.2),
      rightEar: new Point(cx - box.width * 0.4, cy),
      leftEar: new Point(cx + box.width * 0.4, cy),
    };
  }

  /**
   * Dispose of resources.
   */
  public override dispose(throwOnRedispose = true): void {
    // Dispose graph model if loaded
    if (this._graphModel) {
      this._graphModel.dispose();
      this._graphModel = null;
    }
    // Dispose layers model if loaded
    if (this.params) {
      super.dispose(throwOnRedispose);
    }
    // Dispose fallback
    if (this._fallbackNet) {
      this._fallbackNet.dispose(throwOnRedispose);
      this._fallbackNet = null;
    }
  }

  protected getDefaultModelName(): string {
    return 'blazeface_model';
  }

  protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeightMap(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights);
  }
}

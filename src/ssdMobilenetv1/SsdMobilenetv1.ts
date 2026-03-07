import * as tf from '@tensorflow/tfjs';

import { Rect } from '../classes/index';
import { INPUT_RANGES, validateInputRange } from '../common/inputValidation';
import { FaceDetection } from '../classes/FaceDetection';
import type { NetInput, TNetInput } from '../dom/index';
import { toNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { extractParams } from './extractParams';
import { extractParamsFromWeightMap } from './extractParamsFromWeightMap';
import { mobileNetV1 } from './mobileNetV1';
import { nonMaxSuppression } from './nonMaxSuppression';
import { outputLayer } from './outputLayer';
import { predictionLayer } from './predictionLayer';
import type { ISsdMobilenetv1Options } from './SsdMobilenetv1Options';
import { SsdMobilenetv1Options } from './SsdMobilenetv1Options';
import type { NetParams } from './types';

/** Whether to validate input range (can be disabled for performance in production) */
let inputValidationEnabled = process.env.NODE_ENV !== 'production';

/**
 * Enable or disable input validation for SsdMobilenetv1.
 * Validation is enabled by default in development, disabled in production.
 */
export function setInputValidation(enabled: boolean): void {
  inputValidationEnabled = enabled;
}

export class SsdMobilenetv1 extends NeuralNetwork<NetParams> {
  constructor() {
    super('SsdMobilenetv1');
  }

  public forwardInput(input: NetInput) {
    const { params } = this;
    if (!params) throw new Error('SsdMobilenetv1 - load model before inference');
    return tf.tidy(() => {
      const batchTensor = tf.cast(input.toBatchTensor(512, false), 'float32');
      const x = tf.sub(tf.div(batchTensor, 127.5), 1) as tf.Tensor4D; // input is normalized -1..1
      const features = mobileNetV1(x, params.mobilenetv1);
      const { boxPredictions, classPredictions } = predictionLayer(features.out, features.conv11, params.prediction_layer);
      return outputLayer(boxPredictions, classPredictions, params.output_layer);
    });
  }

  /**
   * Forward pass with optional input validation.
   * @param input The input tensor
   * @param validateInput Whether to validate input range (default: based on environment)
   */
  public async forwardInputWithValidation(input: NetInput, validateInput = inputValidationEnabled) {
    if (validateInput) {
      const inputTensor = input.getInput(0);
      if (inputTensor instanceof tf.Tensor) {
        await validateInputRange(inputTensor, INPUT_RANGES.IMAGE_UINT8, {
          modelName: 'SsdMobilenetv1',
          logWarnings: true,
        });
      }
    }
    return this.forwardInput(input);
  }

  public async forward(input: TNetInput) {
    return this.forwardInput(await toNetInput(input));
  }

  public async locateFaces(input: TNetInput, options: ISsdMobilenetv1Options = {}): Promise<FaceDetection[]> {
    const { maxResults, minConfidence } = new SsdMobilenetv1Options(options);
    const netInput = await toNetInput(input);

    const { boxes: _boxes, scores: _scores } = this.forwardInput(netInput);

    // Keep only first batch, dispose the rest immediately
    const boxes = _boxes[0];
    const scores = _scores[0];

    if (!boxes || !scores) {
      throw new Error('SsdMobilenetv1 - no boxes or scores returned from model');
    }

    for (let i = 1; i < _boxes.length; i++) {
      const b = _boxes[i];
      const s = _scores[i];
      if (b) b.dispose();
      if (s) s.dispose();
    }

    try {
      // Use async data() instead of blocking dataSync() for better GPU pipelining
      const [scoresData, boxesData] = await Promise.all([
        scores.data(),
        boxes.array() as Promise<number[][]>,
      ]);

      // Defensive null checks for tensor data
      if (!scoresData || scoresData.length === 0) {
        console.warn('SsdMobilenetv1: No scores data returned from model');
        return [];
      }
      if (!boxesData || boxesData.length === 0) {
        console.warn('SsdMobilenetv1: No boxes data returned from model');
        return [];
      }

      const iouThreshold = 0.5;
      const scoresArray = Array.from(scoresData);
      // Pass pre-fetched boxesData array to avoid GPU blocking in NMS
      const indices = nonMaxSuppression(boxesData, scoresArray, maxResults, iouThreshold, minConfidence);

      const reshapedDims = netInput.getReshapedInputDimensions(0);
      const inputSize = netInput.inputSize as number;
      const padX = inputSize / reshapedDims.width;
      const padY = inputSize / reshapedDims.height;

      const results: FaceDetection[] = [];
      for (const idx of indices) {
        const boxData = boxesData[idx];
        if (!boxData) continue;

        const box0 = boxData[0] ?? 0;
        const box1 = boxData[1] ?? 0;
        const box2 = boxData[2] ?? 0;
        const box3 = boxData[3] ?? 0;

        const top = Math.max(0, box0) * padY;
        const bottom = Math.min(1.0, box2) * padY;
        const left = Math.max(0, box1) * padX;
        const right = Math.min(1.0, box3) * padX;

        const score = scoresArray[idx] ?? 0;
        results.push(new FaceDetection(
          score,
          new Rect(left, top, right - left, bottom - top),
          { height: netInput.getInputHeight(0), width: netInput.getInputWidth(0) },
        ));
      }

      return results;
    } finally {
      // Ensure tensors are disposed even if an error occurs
      boxes.dispose();
      scores.dispose();
    }
  }

  protected getDefaultModelName(): string {
    return 'ssd_mobilenetv1_model';
  }

  protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeightMap(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights);
  }
}

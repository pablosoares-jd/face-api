import * as tf from '../../dist/tfjs.esm';

import { Rect } from '../classes/index';
import { FaceDetection } from '../classes/FaceDetection';
import { NetInput, TNetInput, toNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { extractParams } from './extractParams';
import { extractParamsFromWeightMap } from './extractParamsFromWeightMap';
import { mobileNetV1 } from './mobileNetV1';
import { nonMaxSuppression } from './nonMaxSuppression';
import { outputLayer } from './outputLayer';
import { predictionLayer } from './predictionLayer';
import { ISsdMobilenetv1Options, SsdMobilenetv1Options } from './SsdMobilenetv1Options';
import { NetParams } from './types';

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
    for (let i = 1; i < _boxes.length; i++) {
      _boxes[i].dispose();
      _scores[i].dispose();
    }

    try {
      // Use async data() instead of blocking dataSync() for better GPU pipelining
      const [scoresData, boxesData] = await Promise.all([
        scores.data(),
        boxes.array() as Promise<number[][]>,
      ]);

      const iouThreshold = 0.5;
      const scoresArray = Array.from(scoresData);
      const indices = nonMaxSuppression(boxes, scoresArray, maxResults, iouThreshold, minConfidence);

      const reshapedDims = netInput.getReshapedInputDimensions(0);
      const inputSize = netInput.inputSize as number;
      const padX = inputSize / reshapedDims.width;
      const padY = inputSize / reshapedDims.height;

      const results: FaceDetection[] = [];
      for (const idx of indices) {
        const boxData = boxesData[idx];
        if (!boxData) continue;

        const [top, bottom] = [
          Math.max(0, boxData[0]),
          Math.min(1.0, boxData[2]),
        ].map((val) => val * padY);
        const [left, right] = [
          Math.max(0, boxData[1]),
          Math.min(1.0, boxData[3]),
        ].map((val) => val * padX);

        results.push(new FaceDetection(
          scoresArray[idx],
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

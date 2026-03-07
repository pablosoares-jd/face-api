import * as tf from '@tensorflow/tfjs';

import { BoundingBox } from '../classes/BoundingBox';
import type { Dimensions } from '../classes/Dimensions';
import { ObjectDetection } from '../classes/ObjectDetection';
import { convLayer } from '../common/index';
import type { ConvParams, SeparableConvParams } from '../common/types';
import { toNetInput } from '../dom/index';
import type { NetInput } from '../dom/NetInput';
import type { TNetInput } from '../dom/types';
import { NeuralNetwork } from '../NeuralNetwork';
import { sigmoid } from '../ops/index';
import { nonMaxSuppression } from '../ops/nonMaxSuppression';
import { normalize } from '../ops/normalize';
import type { TinyYolov2Config } from './config';
import { validateConfig } from './config';
import { convWithBatchNorm } from './convWithBatchNorm';
import { depthwiseSeparableConv } from './depthwiseSeparableConv';
import { extractParams } from './extractParams';
import { extractParamsFromWeightMap } from './extractParamsFromWeightMap';
import { leaky } from './leaky';
import type { ITinyYolov2Options } from './TinyYolov2Options';
import { TinyYolov2Options } from './TinyYolov2Options';
import type { DefaultTinyYolov2NetParams, MobilenetParams, TinyYolov2ExtractBoxesResult, TinyYolov2NetParams } from './types';

export class TinyYolov2Base extends NeuralNetwork<TinyYolov2NetParams> {
  public static DEFAULT_FILTER_SIZES = [3, 16, 32, 64, 128, 256, 512, 1024, 1024];

  private _config: TinyYolov2Config;

  constructor(config: TinyYolov2Config) {
    super('TinyYolov2');
    validateConfig(config);
    this._config = config;
  }

  public get config(): TinyYolov2Config {
    return this._config;
  }

  public get withClassScores(): boolean {
    return this.config.withClassScores || this.config.classes.length > 1;
  }

  public get boxEncodingSize(): number {
    return 5 + (this.withClassScores ? this.config.classes.length : 0);
  }

  public runTinyYolov2(x: tf.Tensor4D, params: DefaultTinyYolov2NetParams): tf.Tensor4D {
    let out = convWithBatchNorm(x, params.conv0);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = convWithBatchNorm(out, params.conv1);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = convWithBatchNorm(out, params.conv2);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = convWithBatchNorm(out, params.conv3);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = convWithBatchNorm(out, params.conv4);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = convWithBatchNorm(out, params.conv5);
    out = tf.maxPool(out, [2, 2], [1, 1], 'same');
    out = convWithBatchNorm(out, params.conv6);
    out = convWithBatchNorm(out, params.conv7);
    return convLayer(out, params.conv8, 'valid', false);
  }

  public runMobilenet(x: tf.Tensor4D, params: MobilenetParams): tf.Tensor4D {
    let out = this.config.isFirstLayerConv2d
      ? leaky(convLayer(x, params.conv0 as ConvParams, 'valid', false))
      : depthwiseSeparableConv(x, params.conv0 as SeparableConvParams);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = depthwiseSeparableConv(out, params.conv1);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = depthwiseSeparableConv(out, params.conv2);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = depthwiseSeparableConv(out, params.conv3);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = depthwiseSeparableConv(out, params.conv4);
    out = tf.maxPool(out, [2, 2], [2, 2], 'same');
    out = depthwiseSeparableConv(out, params.conv5);
    out = tf.maxPool(out, [2, 2], [1, 1], 'same');
    out = params.conv6 ? depthwiseSeparableConv(out, params.conv6) : out;
    out = params.conv7 ? depthwiseSeparableConv(out, params.conv7) : out;
    return convLayer(out, params.conv8, 'valid', false);
  }

  public forwardInput(input: NetInput, inputSize: number): tf.Tensor4D {
    const { params } = this;

    if (!params) {
      throw new Error('TinyYolov2 - load model before inference');
    }

    return tf.tidy(() => {
      let batchTensor = tf.cast(input.toBatchTensor(inputSize, false), 'float32');
      batchTensor = this.config.meanRgb
        ? normalize(batchTensor, this.config.meanRgb)
        : batchTensor;
      batchTensor = batchTensor.div(255) as tf.Tensor4D;
      return this.config.withSeparableConvs
        ? this.runMobilenet(batchTensor, params as MobilenetParams)
        : this.runTinyYolov2(batchTensor, params as DefaultTinyYolov2NetParams);
    });
  }

  public async forward(input: TNetInput, inputSize: number): Promise<tf.Tensor4D> {
    return this.forwardInput(await toNetInput(input), inputSize);
  }

  public async detect(input: TNetInput, forwardParams: ITinyYolov2Options = {}): Promise<ObjectDetection[]> {
    const { inputSize, scoreThreshold } = new TinyYolov2Options(forwardParams);
    const netInput = await toNetInput(input);
    const out = await this.forwardInput(netInput, inputSize);
    const out0 = tf.tidy(() => {
      const unstacked = tf.unstack(out)[0];
      if (!unstacked) {
        throw new Error('TinyYolov2Base.detect - failed to unstack output tensor');
      }
      return unstacked.expandDims();
    }) as tf.Tensor4D;
    const inputDimensions = {
      width: netInput.getInputWidth(0),
      height: netInput.getInputHeight(0),
    };

    const results = await this.extractBoxes(out0, netInput.getReshapedInputDimensions(0), scoreThreshold);
    out.dispose();
    out0.dispose();

    const boxes = results.map((res) => res.box);
    const scores = results.map((res) => res.score);
    const classScores = results.map((res) => res.classScore);
    const classNames = results.map((res) => this.config.classes[res.label]);

    const indices = nonMaxSuppression(
      boxes.map((box) => box.rescale(inputSize)),
      scores,
      this.config.iouThreshold,
      true,
    );

    const detections = indices.map((idx) => {
      const score = scores[idx];
      const classScore = classScores[idx];
      const className = classNames[idx];
      const box = boxes[idx];
      if (score === undefined || classScore === undefined || className === undefined || box === undefined) {
        throw new Error(`TinyYolov2Base.detect - invalid result at index ${idx}`);
      }
      return new ObjectDetection(score, classScore, className, box, inputDimensions);
    });
    return detections;
  }

  protected getDefaultModelName(): string {
    return '';
  }

  protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeightMap(weightMap, this.config);
  }

  protected extractParams(weights: Float32Array) {
    const filterSizes = this.config.filterSizes || TinyYolov2Base.DEFAULT_FILTER_SIZES;

    const numFilters = filterSizes ? filterSizes.length : undefined;
    if (numFilters !== 7 && numFilters !== 8 && numFilters !== 9) {
      throw new Error(`TinyYolov2 - expected 7 | 8 | 9 convolutional filters, but found ${numFilters} filterSizes in config`);
    }
    return extractParams(weights, this.config, this.boxEncodingSize, filterSizes);
  }

  protected async extractBoxes(
    outputTensor: tf.Tensor4D,
    inputBlobDimensions: Dimensions,
    scoreThreshold?: number,
  ) {
    const { width, height } = inputBlobDimensions;
    const inputSize = Math.max(width, height);
    const correctionFactorX = inputSize / width;
    const correctionFactorY = inputSize / height;

    const numCells = outputTensor.shape[1];
    const numBoxes = this.config.anchors.length;

    const [boxesTensor, scoresTensor, classScoresTensor] = tf.tidy(() => {
      const reshaped = outputTensor.reshape([numCells, numCells, numBoxes, this.boxEncodingSize]);

      const boxes = reshaped.slice([0, 0, 0, 0], [numCells, numCells, numBoxes, 4]);
      const scores = reshaped.slice([0, 0, 0, 4], [numCells, numCells, numBoxes, 1]);
      const classScores = this.withClassScores
        ? tf.softmax(reshaped.slice([0, 0, 0, 5], [numCells, numCells, numBoxes, this.config.classes.length]), 3)
        : tf.scalar(0);
      return [boxes, scores, classScores];
    });

    try {
      // Fetch all data at once instead of inside loops - major performance improvement
      const [scoresData, boxesData, classScoresData] = await Promise.all([
        scoresTensor.array() as Promise<number[][][][]>,
        boxesTensor.array() as Promise<number[][][][]>,
        this.withClassScores ? (classScoresTensor as tf.Tensor4D).array() as Promise<number[][][][]> : Promise.resolve(null),
      ]);

      const results: TinyYolov2ExtractBoxesResult[] = [];

      // Process all cells without async/await inside loops
      for (let row = 0; row < numCells; row++) {
        const scoreRow = scoresData[row];
        const boxRow = boxesData[row];
        if (!scoreRow || !boxRow) continue;

        for (let col = 0; col < numCells; col++) {
          const scoreCol = scoreRow[col];
          const boxCol = boxRow[col];
          if (!scoreCol || !boxCol) continue;

          for (let anchor = 0; anchor < numBoxes; anchor++) {
            const scoreAnchor = scoreCol[anchor];
            const boxData = boxCol[anchor];
            const anchorConfig = this.config.anchors[anchor];
            if (!scoreAnchor || !boxData || !anchorConfig) continue;

            const scoreVal = scoreAnchor[0];
            if (scoreVal === undefined) continue;
            const score = sigmoid(scoreVal);

            if (!scoreThreshold || score > scoreThreshold) {
              const box0 = boxData[0] ?? 0;
              const box1 = boxData[1] ?? 0;
              const box2 = boxData[2] ?? 0;
              const box3 = boxData[3] ?? 0;

              const ctX = ((col + sigmoid(box0)) / numCells) * correctionFactorX;
              const ctY = ((row + sigmoid(box1)) / numCells) * correctionFactorY;
              const widthLocal = ((Math.exp(box2) * anchorConfig.x) / numCells) * correctionFactorX;
              const heightLocal = ((Math.exp(box3) * anchorConfig.y) / numCells) * correctionFactorY;
              const x = ctX - (widthLocal / 2);
              const y = ctY - (heightLocal / 2);

              // Extract class scores synchronously from pre-fetched data
              let classScore = 1;
              let labelVal = 0;

              if (this.withClassScores && classScoresData) {
                const classRow = classScoresData[row];
                const classCol = classRow?.[col];
                const classData = classCol?.[anchor];
                if (classData) {
                  for (let i = 0; i < this.config.classes.length; i++) {
                    const classVal = classData[i] ?? 0;
                    if (classVal > classScore || i === 0) {
                      classScore = classVal;
                      labelVal = i;
                    }
                  }
                }
              }

              results.push({
                box: new BoundingBox(x, y, x + widthLocal, y + heightLocal),
                score,
                classScore: score * classScore,
                label: labelVal,
                row,
                col,
                anchor,
              });
            }
          }
        }
      }

      return results;
    } finally {
      // Ensure tensors are disposed even if an error occurs
      boxesTensor.dispose();
      scoresTensor.dispose();
      classScoresTensor.dispose();
    }
  }

  // extractPredictedClass is available for subclass implementations
  protected async extractPredictedClass(classesTensor: tf.Tensor4D, pos: { row: number, col: number, anchor: number }) {
    const { row, col, anchor } = pos;
    const classesData = await classesTensor.array() as number[][][][];
    const rowData = classesData[row];
    const colData = rowData?.[col];
    const classData = colData?.[anchor];

    if (!classData) {
      return { classScore: 0, label: 0 };
    }

    let maxScore = classData[0] ?? 0;
    let maxLabel = 0;

    for (let i = 1; i < this.config.classes.length; i++) {
      const score = classData[i] ?? 0;
      if (score > maxScore) {
        maxScore = score;
        maxLabel = i;
      }
    }

    return { classScore: maxScore, label: maxLabel };
  }
}

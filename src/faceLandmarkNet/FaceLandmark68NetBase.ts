import * as tf from '@tensorflow/tfjs';

import { IDimensions, Point } from '../classes/index';
import { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import { NetInput, TNetInput, toNetInput } from '../dom/index';
import { FaceFeatureExtractorParams, TinyFaceFeatureExtractorParams } from '../faceFeatureExtractor/types';
import { FaceProcessor } from '../faceProcessor/FaceProcessor';
import { isEven } from '../utils/index';

export abstract class FaceLandmark68NetBase<
  TExtractorParams extends FaceFeatureExtractorParams | TinyFaceFeatureExtractorParams
>
  extends FaceProcessor<TExtractorParams> {
  public postProcess(output: tf.Tensor2D, inputSize: number, originalDimensions: IDimensions[]): tf.Tensor2D {
    const inputDimensions = originalDimensions.map(({ width, height }) => {
      const scale = inputSize / Math.max(height, width);
      return {
        width: width * scale,
        height: height * scale,
      };
    });

    const batchSize = inputDimensions.length;

    return tf.tidy(() => {
      // Pre-compute padding and dimension arrays for vectorized operations
      const paddingsX: number[] = [];
      const paddingsY: number[] = [];
      const widths: number[] = [];
      const heights: number[] = [];

      for (let i = 0; i < batchSize; i++) {
        const { width, height } = inputDimensions[i];
        paddingsX.push(width < height ? Math.abs(width - height) / 2 : 0);
        paddingsY.push(height < width ? Math.abs(width - height) / 2 : 0);
        widths.push(width);
        heights.push(height);
      }

      // Create interleaved tensors more efficiently using tf.tensor2d
      const createBatchInterleavedTensor = (xVals: number[], yVals: number[]) => {
        const data: number[][] = [];
        for (let b = 0; b < batchSize; b++) {
          const row: number[] = [];
          for (let i = 0; i < 68; i++) {
            row.push(xVals[b], yVals[b]);
          }
          data.push(row);
        }
        return tf.tensor2d(data, [batchSize, 136], 'float32');
      };

      // Vectorized computation: (output * inputSize - padding) / dimensions
      const scaled = output.mul(tf.scalar(inputSize, 'float32'));
      const paddingTensor = createBatchInterleavedTensor(paddingsX, paddingsY);
      const dimTensor = createBatchInterleavedTensor(widths, heights);

      return scaled.sub(paddingTensor).div(dimTensor) as tf.Tensor2D;
    });
  }

  public forwardInput(input: NetInput): tf.Tensor2D {
    return tf.tidy(() => {
      const out = this.runNet(input);
      return this.postProcess(
        out,
        input.inputSize as number,
        input.inputDimensions.map(([height, width]) => ({ height, width })),
      );
    });
  }

  public async forward(input: TNetInput): Promise<tf.Tensor2D> {
    return this.forwardInput(await toNetInput(input));
  }

  public async detectLandmarks(input: TNetInput): Promise<FaceLandmarks68 | FaceLandmarks68[]> {
    const netInput = await toNetInput(input);
    const landmarkTensors = tf.tidy(
      () => tf.unstack(this.forwardInput(netInput)),
    );

    try {
      // Use async data() for better GPU pipelining instead of blocking dataSync()
      const landmarkDataPromises = landmarkTensors.map((t) => t.data());
      const landmarkDataArrays = await Promise.all(landmarkDataPromises);

      const landmarksForBatch = landmarkDataArrays.map((landmarksData, batchIdx) => {
        // Extract x and y coordinates more efficiently
        const points: Point[] = new Array(68);
        for (let i = 0; i < 68; i++) {
          points[i] = new Point(
            landmarksData[i * 2] as number,
            landmarksData[i * 2 + 1] as number
          );
        }

        return new FaceLandmarks68(points, {
          height: netInput.getInputHeight(batchIdx),
          width: netInput.getInputWidth(batchIdx),
        });
      });

      return netInput.isBatchInput ? landmarksForBatch : landmarksForBatch[0];
    } finally {
      // Ensure tensors are disposed even if an error occurs
      landmarkTensors.forEach((t) => t.dispose());
    }
  }

  protected getClassifierChannelsOut(): number {
    return 136;
  }
}

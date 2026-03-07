import * as tf from '@tensorflow/tfjs';

import type { NetInput, TNetInput } from '../dom/index';
import { toNetInput } from '../dom/index';
import { FaceFeatureExtractor } from '../faceFeatureExtractor/FaceFeatureExtractor';
import type { FaceFeatureExtractorParams } from '../faceFeatureExtractor/types';
import { FaceProcessor } from '../faceProcessor/FaceProcessor';
import { FaceExpressions } from './FaceExpressions';

export class FaceExpressionNet extends FaceProcessor<FaceFeatureExtractorParams> {
  constructor(faceFeatureExtractor: FaceFeatureExtractor = new FaceFeatureExtractor()) {
    super('FaceExpressionNet', faceFeatureExtractor);
  }

  public forwardInput(input: NetInput | tf.Tensor4D): tf.Tensor2D {
    return tf.tidy(() => tf.softmax(this.runNet(input)));
  }

  public async forward(input: TNetInput): Promise<tf.Tensor2D> {
    return this.forwardInput(await toNetInput(input));
  }

  public async predictExpressions(input: TNetInput): Promise<FaceExpressions | FaceExpressions[]> {
    const netInput = await toNetInput(input);
    const out = await this.forwardInput(netInput);

    // Unstack inside tidy to avoid memory leaks if Promise.all fails
    const tensors = tf.unstack(out);

    try {
      // Use async data() instead of blocking dataSync() for better GPU pipelining
      const dataPromises = tensors.map((t) => t.data());
      const probabilitesByBatch = await Promise.all(dataPromises);

      const predictionsByBatch = probabilitesByBatch
        .map((probabilites) => new FaceExpressions(probabilites as Float32Array));

      if (netInput.isBatchInput) {
        return predictionsByBatch;
      }
      const firstPrediction = predictionsByBatch[0];
      if (!firstPrediction) {
        throw new Error('FaceExpressionNet.predictExpressions - no predictions generated');
      }
      return firstPrediction;
    } finally {
      // Ensure tensors are disposed even if an error occurs
      tensors.forEach((t) => t.dispose());
      out.dispose();
    }
  }

  protected getDefaultModelName(): string {
    return 'face_expression_model';
  }

  protected getClassifierChannelsIn(): number {
    return 256;
  }

  protected getClassifierChannelsOut(): number {
    return 7;
  }
}

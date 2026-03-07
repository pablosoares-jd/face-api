import * as tf from '@tensorflow/tfjs';

import { INPUT_RANGES, validateInputRange } from '../common/inputValidation';
import type { NetInput, TNetInput } from '../dom/index';
import { toNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { normalize } from '../ops/index';
import { convDown } from './convLayer';
import { extractParams } from './extractParams';
import { extractParamsFromWeightMap } from './extractParamsFromWeightMap';
import { residual, residualDown } from './residualLayer';
import type { NetParams } from './types';

/** Whether to validate input range (can be disabled for performance in production) */
let inputValidationEnabled = process.env.NODE_ENV !== 'production';

/**
 * Enable or disable input validation for FaceRecognitionNet.
 * Validation is enabled by default in development, disabled in production.
 */
export function setFaceRecognitionInputValidation(enabled: boolean): void {
  inputValidationEnabled = enabled;
}

export class FaceRecognitionNet extends NeuralNetwork<NetParams> {
  constructor() {
    super('FaceRecognitionNet');
  }

  public forwardInput(input: NetInput): tf.Tensor2D {
    const { params } = this;

    if (!params) {
      throw new Error('FaceRecognitionNet - load model before inference');
    }

    return tf.tidy(() => {
      const batchTensor = tf.cast(input.toBatchTensor(150, true), 'float32');

      const meanRgb = [122.782, 117.001, 104.298];
      const normalized = normalize(batchTensor, meanRgb).div(255) as tf.Tensor4D;

      let out = convDown(normalized, params.conv32_down);
      out = tf.maxPool(out, 3, 2, 'valid');

      out = residual(out, params.conv32_1);
      out = residual(out, params.conv32_2);
      out = residual(out, params.conv32_3);

      out = residualDown(out, params.conv64_down);
      out = residual(out, params.conv64_1);
      out = residual(out, params.conv64_2);
      out = residual(out, params.conv64_3);

      out = residualDown(out, params.conv128_down);
      out = residual(out, params.conv128_1);
      out = residual(out, params.conv128_2);

      out = residualDown(out, params.conv256_down);
      out = residual(out, params.conv256_1);
      out = residual(out, params.conv256_2);
      out = residualDown(out, params.conv256_down_out);

      const globalAvg = out.mean([1, 2]) as tf.Tensor2D;
      const fullyConnected = tf.matMul(globalAvg, params.fc);

      return fullyConnected as tf.Tensor2D;
    });
  }

  public async forward(input: TNetInput): Promise<tf.Tensor2D> {
    return this.forwardInput(await toNetInput(input));
  }

  public async computeFaceDescriptor(input: TNetInput, validateInput = inputValidationEnabled): Promise<Float32Array|Float32Array[]> {
    // Check for invalid tensor input shape
    if (input && typeof input === 'object' && 'shape' in input) {
      const tensorInput = input as tf.Tensor;
      if (tensorInput.shape?.some((dim) => dim <= 0)) {
        return new Float32Array(128);
      }
    }

    const netInput = await toNetInput(input);

    // Validate input range if enabled
    if (validateInput) {
      const inputTensor = netInput.getInput(0);
      if (inputTensor instanceof tf.Tensor) {
        await validateInputRange(inputTensor, INPUT_RANGES.IMAGE_UINT8, {
          modelName: 'FaceRecognitionNet',
          logWarnings: true,
        });
      }
    }

    const faceDescriptorTensors = tf.tidy(() => tf.unstack(this.forwardInput(netInput)));

    // Defensive check: if no tensors were produced, return empty descriptor
    if (!faceDescriptorTensors || faceDescriptorTensors.length === 0) {
      console.warn('FaceRecognitionNet: No face descriptor tensors produced');
      return new Float32Array(128);
    }

    const faceDescriptorsForBatch = await Promise.all(faceDescriptorTensors.map((t) => t.data())) as Float32Array[];
    faceDescriptorTensors.forEach((t) => t.dispose());

    // Defensive check: if batch is empty, return empty descriptor
    if (!faceDescriptorsForBatch || faceDescriptorsForBatch.length === 0) {
      console.warn('FaceRecognitionNet: No face descriptors computed');
      return new Float32Array(128);
    }

    // For non-batch input, ensure we have a valid descriptor
    if (!netInput.isBatchInput) {
      const descriptor = faceDescriptorsForBatch[0];
      if (!descriptor) {
        console.warn('FaceRecognitionNet: First descriptor is null/undefined');
        return new Float32Array(128);
      }
      return descriptor;
    }

    return faceDescriptorsForBatch;
  }

  protected getDefaultModelName(): string {
    return 'face_recognition_model';
  }

  protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeightMap(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights);
  }
}

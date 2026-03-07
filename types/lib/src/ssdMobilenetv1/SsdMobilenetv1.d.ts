import * as tf from '@tensorflow/tfjs';
import { FaceDetection } from '../classes/FaceDetection';
import type { NetInput, TNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import type { ISsdMobilenetv1Options } from './SsdMobilenetv1Options';
import type { NetParams } from './types';
/**
 * Enable or disable input validation for SsdMobilenetv1.
 * Validation is enabled by default in development, disabled in production.
 */
export declare function setInputValidation(enabled: boolean): void;
export declare class SsdMobilenetv1 extends NeuralNetwork<NetParams> {
    constructor();
    forwardInput(input: NetInput): {
        boxes: tf.Tensor2D[];
        scores: tf.Tensor1D[];
    };
    /**
     * Forward pass with optional input validation.
     * @param input The input tensor
     * @param validateInput Whether to validate input range (default: based on environment)
     */
    forwardInputWithValidation(input: NetInput, validateInput?: boolean): Promise<{
        boxes: tf.Tensor2D[];
        scores: tf.Tensor1D[];
    }>;
    forward(input: TNetInput): Promise<{
        boxes: tf.Tensor2D[];
        scores: tf.Tensor1D[];
    }>;
    locateFaces(input: TNetInput, options?: ISsdMobilenetv1Options): Promise<FaceDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: NetParams;
        paramMappings: import("../common").ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams;
        paramMappings: import("../common").ParamMapping[];
    };
}

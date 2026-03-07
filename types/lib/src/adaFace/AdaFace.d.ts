import * as tf from '@tensorflow/tfjs';
import { NetInput, TNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { FaceRecognitionNet } from '../faceRecognitionNet/FaceRecognitionNet';
import { IAdaFaceOptions } from './AdaFaceOptions';
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
export declare class AdaFace extends NeuralNetwork<NetParams> {
    private _fallbackNet;
    constructor();
    /**
     * Get the fallback recognizer (FaceNet/FaceRecognitionNet).
     */
    get fallbackNet(): FaceRecognitionNet | null;
    /**
     * Load AdaFace model with optional fallback to FaceNet.
     */
    load(weightsOrUrl: Float32Array | string | undefined): Promise<void>;
    /**
     * Load fallback recognizer explicitly.
     */
    loadFallback(weightsOrUrl: Float32Array | string | undefined): Promise<void>;
    /**
     * Check if the primary model is loaded.
     */
    get isPrimaryLoaded(): boolean;
    /**
     * Check if any model is available for inference.
     */
    get isLoaded(): boolean;
    /**
     * Forward pass through the network.
     */
    forwardInput(input: NetInput): tf.Tensor2D;
    /**
     * Convolution + BatchNorm + ReLU block.
     */
    private convBnRelu;
    /**
     * IR-SE (Inverted Residual with Squeeze-Excitation) block.
     */
    private irSeBlock;
    /**
     * Squeeze-Excitation module.
     */
    private squeezeExcitation;
    /**
     * Compute face descriptor (embedding) for recognition.
     */
    computeFaceDescriptor(input: TNetInput, options?: IAdaFaceOptions): Promise<Float32Array | Float32Array[]>;
    /**
     * Blend two face descriptors with weighted average.
     * Handles different descriptor sizes by truncating to minimum length.
     */
    private blendDescriptors;
    forward(input: TNetInput): Promise<tf.Tensor2D>;
    /**
     * Dispose of resources.
     */
    dispose(throwOnRedispose?: boolean): void;
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

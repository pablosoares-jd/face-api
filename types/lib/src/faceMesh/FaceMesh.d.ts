import * as tf from '@tensorflow/tfjs';
import { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import { NetInput, TNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { FaceLandmark68Net } from '../faceLandmarkNet/FaceLandmark68Net';
import { IFaceMeshOptions } from './FaceMeshOptions';
import { FaceMeshLandmarks } from './FaceMeshLandmarks';
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
export declare class FaceMesh extends NeuralNetwork<NetParams> {
    private _fallbackNet;
    constructor();
    /**
     * Get the fallback detector (68-point landmarks).
     */
    get fallbackNet(): FaceLandmark68Net | null;
    /**
     * Load FaceMesh model with optional fallback to 68-point landmarks.
     */
    load(weightsOrUrl: Float32Array | string | undefined): Promise<void>;
    /**
     * Load fallback detector explicitly.
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
     * Convolution block.
     */
    private convBlock;
    /**
     * Residual block.
     */
    private residualBlock;
    /**
     * Detect facial landmarks.
     * @param input Input image
     * @param options Detection options including refineLandmarks for iris detection
     */
    detectLandmarks(input: TNetInput, options?: IFaceMeshOptions): Promise<FaceMeshLandmarks | FaceMeshLandmarks[] | FaceLandmarks68 | FaceLandmarks68[]>;
    /**
     * Estimate iris landmarks (468-477) from eye landmarks.
     * This provides approximate iris positions when a refined model is not available.
     */
    private estimateIrisLandmarks;
    /**
     * Detect landmarks and convert to 68-point format for compatibility.
     */
    detectLandmarks68(input: TNetInput, options?: IFaceMeshOptions): Promise<FaceLandmarks68 | FaceLandmarks68[]>;
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

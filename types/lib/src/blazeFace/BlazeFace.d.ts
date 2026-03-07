import * as tf from '@tensorflow/tfjs';
import { Point, Rect } from '../classes/index';
import { FaceDetection } from '../classes/FaceDetection';
import { NetInput, TNetInput } from '../dom/index';
import { NeuralNetwork } from '../NeuralNetwork';
import { SsdMobilenetv1 } from '../ssdMobilenetv1/SsdMobilenetv1';
import { IBlazeFaceOptions } from './BlazeFaceOptions';
import { NetParams } from './types';
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
export declare class BlazeFaceDetection extends FaceDetection {
    private _keypoints;
    constructor(score: number, relativeBox: Rect, imageDims: {
        width: number;
        height: number;
    }, keypoints: BlazeFaceKeypoints);
    /**
     * Get the 6 facial keypoints.
     */
    get keypoints(): BlazeFaceKeypoints;
    /**
     * Get keypoints as an array of Points.
     */
    get keypointsArray(): Point[];
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
export declare class BlazeFace extends NeuralNetwork<NetParams> {
    private _fallbackNet;
    private _graphModel;
    private _anchors;
    private _currentInputSize;
    private static readonly STRIDES;
    private static readonly ANCHORS_PER_STRIDE;
    constructor();
    /**
     * Check if graph model is loaded (for graph-model format).
     */
    get isGraphModelLoaded(): boolean;
    /**
     * Get the fallback detector (SSD MobileNetv1).
     */
    get fallbackNet(): SsdMobilenetv1 | null;
    /**
     * Generate anchors for the model.
     */
    private generateAnchors;
    /**
     * Load BlazeFace model with optional fallback to SSD MobileNetv1.
     * Tries graph-model format first, then layers-model, then falls back to SSD.
     */
    load(weightsOrUrl: Float32Array | string | undefined): Promise<void>;
    /**
     * Load fallback detector explicitly.
     */
    loadFallback(weightsOrUrl: Float32Array | string | undefined): Promise<void>;
    /**
     * Check if the primary model is loaded (either graph-model or layers-model).
     */
    get isPrimaryLoaded(): boolean;
    /**
     * Check if any model is available for inference.
     */
    get isLoaded(): boolean;
    /**
     * Check which model type is loaded.
     */
    get modelType(): 'graph' | 'layers' | 'fallback' | 'none';
    /**
     * Forward pass through the graph model.
     * Graph model output format: [batch, num_detections, 17]
     * - 4 values for bounding box (center_x, center_y, width, height)
     * - 1 value for score
     * - 12 values for keypoints (6 points × 2 coords)
     */
    private forwardGraphModel;
    /**
     * Forward pass through the network.
     * @param input The input tensor
     * @param inputSize The input size (128 or 256)
     * @returns Raw regressor output (16 values per detection) and scores
     */
    forwardInput(input: NetInput, inputSize?: 128 | 256): {
        rawBoxes: tf.Tensor2D;
        scores: tf.Tensor1D;
    };
    /**
     * Convolution block with batch norm and activation.
     */
    private convBlock;
    /**
     * Decoded detection result with box and keypoints.
     */
    private decodeDetections;
    /**
     * Detect faces in an image.
     * Returns BlazeFaceDetection objects with 6 facial keypoints.
     */
    locateFaces(input: TNetInput, options?: IBlazeFaceOptions): Promise<BlazeFaceDetection[]>;
    /**
     * Create default keypoints based on bounding box (for fallback).
     */
    private createDefaultKeypoints;
    /**
     * Non-maximum suppression for detected boxes.
     */
    private nonMaxSuppression;
    /**
     * Calculate IOU between two boxes.
     */
    private calculateIOU;
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

import type { Tensor3D } from '@tensorflow/tfjs';
import type { Tensor4D } from '@tensorflow/tfjs';
import * as tf from '@tensorflow/tfjs';

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
export declare class AdaFace extends NeuralNetwork<NetParams_6> {
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
        params: NetParams_6;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams_6;
        paramMappings: ParamMapping[];
    };
}

export declare class AdaFaceOptions {
    readonly inputSize: number;
    readonly descriptorSize: 128 | 256 | 512;
    readonly enableFallback: boolean;
    readonly qualityThreshold: number;
    readonly blendDescriptors: boolean;
    readonly blendWeight: number;
    constructor(options?: IAdaFaceOptions);
}

/**
 * Quality thresholds that vary by source type.
 */
export declare interface AdaptiveThresholds {
    /** Minimum detection confidence */
    minConfidence: number;
    /** Minimum sharpness */
    minSharpness: number;
    /** Maximum yaw angle */
    maxYaw: number;
    /** Maximum pitch angle */
    maxPitch: number;
    /** Minimum face size */
    minFaceSize: number;
}

export declare type AgeAndGenderPrediction = {
    age: number;
    gender: Gender;
    genderProbability: number;
};

export declare class AgeGenderNet extends NeuralNetwork<NetParams> {
    private _faceFeatureExtractor;
    constructor(faceFeatureExtractor?: TinyXception);
    get faceFeatureExtractor(): TinyXception;
    runNet(input: NetInput | tf.Tensor4D): NetOutput;
    forwardInput(input: NetInput | tf.Tensor4D): NetOutput;
    forward(input: TNetInput): Promise<NetOutput>;
    predictAgeAndGender(input: TNetInput): Promise<AgeAndGenderPrediction | AgeAndGenderPrediction[]>;
    protected getDefaultModelName(): string;
    dispose(throwOnRedispose?: boolean): void;
    loadClassifierParams(weights: Float32Array): void;
    extractClassifierParams(weights: Float32Array): {
        params: NetParams;
        paramMappings: ParamMapping[];
    };
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: NetParams;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams;
        paramMappings: ParamMapping[];
    };
}

export declare const allFaces: typeof allFacesSsdMobilenetv1;

export declare function allFacesSsdMobilenetv1(input: TNetInput, minConfidence?: number): Promise<WithFaceDescriptor<WithFaceLandmarks<WithFaceDetection<{}>>>[]>;

export declare function allFacesTinyYolov2(input: TNetInput, forwardParams?: ITinyYolov2Options): Promise<WithFaceDescriptor<WithFaceLandmarks<WithFaceDetection<{}>>>[]>;

declare enum AnchorPosition {
    TOP_LEFT = "TOP_LEFT",
    TOP_RIGHT = "TOP_RIGHT",
    BOTTOM_LEFT = "BOTTOM_LEFT",
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
}

export declare function awaitMediaLoaded(media: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): Promise<unknown>;

/**
 * Result of backend initialization with fallback.
 */
export declare interface BackendInitResult {
    /** The backend that was successfully initialized */
    backend: BackendType;
    /** Whether this was the preferred backend */
    isPreferred: boolean;
    /** Error message if preferred backend failed */
    fallbackReason?: string;
}

/**
 * Backend performance characteristics.
 */
export declare interface BackendPerformance {
    backend: BackendType;
    estimatedSpeedup: number;
    memoryEfficiency: 'high' | 'medium' | 'low';
    parallelization: 'gpu' | 'simd' | 'none';
}

/**
 * Backend priority order for automatic fallback.
 */
export declare type BackendType = 'webgpu' | 'webgl' | 'wasm' | 'cpu';

export declare type BatchNorm = {
    sub: tf.Tensor1D;
    truediv: tf.Tensor1D;
};

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
export declare class BlazeFace extends NeuralNetwork<NetParams_5> {
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
        params: NetParams_5;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams_5;
        paramMappings: ParamMapping[];
    };
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
 * BlazeFace facial keypoints.
 * These are the 6 keypoints returned by BlazeFace.
 */
export declare interface BlazeFaceKeypoints {
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

export declare class BlazeFaceOptions {
    readonly minConfidence: number;
    readonly maxResults: number;
    readonly inputSize: 128 | 256;
    readonly iouThreshold: number;
    readonly enableFallback: boolean;
    constructor(options?: IBlazeFaceOptions);
}

export declare class BoundingBox extends Box implements IBoundingBox {
    constructor(left: number, top: number, right: number, bottom: number, allowNegativeDimensions?: boolean);
}

export declare class Box<BoxType = any> implements IBoundingBox, IRect {
    static isRect(rect: any): boolean;
    static assertIsValidBox(box: any, callee: string, allowNegativeDimensions?: boolean): void;
    private _x;
    private _y;
    private _width;
    private _height;
    constructor(_box: IBoundingBox | IRect, allowNegativeDimensions?: boolean);
    get x(): number;
    get y(): number;
    get width(): number;
    get height(): number;
    get left(): number;
    get top(): number;
    get right(): number;
    get bottom(): number;
    get area(): number;
    get topLeft(): Point;
    get topRight(): Point;
    get bottomLeft(): Point;
    get bottomRight(): Point;
    round(): Box<BoxType>;
    floor(): Box<BoxType>;
    toSquare(): Box<BoxType>;
    rescale(s: IDimensions | number): Box<BoxType>;
    pad(padX: number, padY: number): Box<BoxType>;
    clipAtImageBorders(imgWidth: number, imgHeight: number): Box<BoxType>;
    shift(sx: number, sy: number): Box<BoxType>;
    padAtBorders(imageHeight: number, imageWidth: number): {
        dy: number;
        edy: number;
        dx: number;
        edx: number;
        y: number;
        ey: number;
        x: number;
        ex: number;
        w: number;
        h: number;
    };
    calibrate(region: Box): Box<any>;
}

declare type BoxPredictionParams = {
    box_encoding_predictor: ConvParams;
    class_predictor: ConvParams;
};

export declare function bufferToImage(buf: Blob): Promise<HTMLImageElement>;

export declare class ComposableTask<T> {
    then(onfulfilled: (value: T) => T | PromiseLike<T>): Promise<T>;
    run(): Promise<T>;
}

export declare class ComputeAllFaceDescriptorsTask<TSource extends WithFaceLandmarks<WithFaceDetection<{}>>> extends ComputeFaceDescriptorsTaskBase<WithFaceDescriptor<TSource>[], TSource[]> {
    run(): Promise<WithFaceDescriptor<TSource>[]>;
    withFaceExpressions(): PredictAllFaceExpressionsWithFaceAlignmentTask<WithFaceDescriptor<TSource>>;
    withAgeAndGender(): PredictAllAgeAndGenderWithFaceAlignmentTask<WithFaceDescriptor<TSource>>;
}

/**
 * Computes a 128 entry vector (face descriptor / face embeddings) from the face shown in an image,
 * which uniquely represents the features of that persons face. The computed face descriptor can
 * be used to measure the similarity between faces, by computing the euclidean distance of two
 * face descriptors.
 *
 * @param inputs The face image extracted from the aligned bounding box of a face. Can
 * also be an array of input images, which will be batch processed.
 * @returns Face descriptor with 128 entries or array thereof in case of batch input.
 */
export declare const computeFaceDescriptor: (input: TNetInput) => Promise<Float32Array | Float32Array[]>;

export declare class ComputeFaceDescriptorsTaskBase<TReturn, TParentReturn> extends ComposableTask<TReturn> {
    protected parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>;
    protected input: TNetInput;
    constructor(parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>, input: TNetInput);
}

declare function computeReshapedDimensions({ width, height }: IDimensions, inputSize: number): Dimensions;

export declare class ComputeSingleFaceDescriptorTask<TSource extends WithFaceLandmarks<WithFaceDetection<{}>>> extends ComputeFaceDescriptorsTaskBase<WithFaceDescriptor<TSource> | undefined, TSource | undefined> {
    run(): Promise<WithFaceDescriptor<TSource> | undefined>;
    withFaceExpressions(): PredictSingleFaceExpressionsWithFaceAlignmentTask<WithFaceDescriptor<TSource>>;
    withAgeAndGender(): PredictSingleAgeAndGenderWithFaceAlignmentTask<WithFaceDescriptor<TSource>>;
}

/**
 * BlazeFace convolution block parameters.
 */
export declare interface ConvBlockParams {
    depthwise: tf.Tensor4D;
    depthwiseBias: tf.Tensor1D;
    pointwise: tf.Tensor4D;
    pointwiseBias: tf.Tensor1D;
}

/**
 * Convolution + BatchNorm parameters.
 */
export declare interface ConvBnParams {
    weights: tf.Tensor4D;
    stride?: number;
    bn_mean: tf.Tensor1D;
    bn_variance: tf.Tensor1D;
    bn_offset: tf.Tensor1D;
    bn_scale: tf.Tensor1D;
}

declare type ConvLayerParams = {
    conv: ConvParams;
    scale: ScaleLayerParams;
};

declare type ConvParams = {
    filters: tf.Tensor4D;
    bias: tf.Tensor1D;
};

export declare type ConvWithBatchNorm = {
    conv: ConvParams;
    bn: BatchNorm;
};

declare function createBrowserEnv(): Environment;

export declare function createCanvas({ width, height }: IDimensions): HTMLCanvasElement;

export declare function createCanvasFromMedia(media: HTMLImageElement | HTMLVideoElement | ImageData, dims?: IDimensions): HTMLCanvasElement;

export declare function createFaceDetectionNet(weights: Float32Array): SsdMobilenetv1;

export declare function createFaceRecognitionNet(weights: Float32Array): FaceRecognitionNet;

declare function createFileSystem(fs?: any): FileSystem_2;

declare function createNodejsEnv(): Environment;

export declare function createSsdMobilenetv1(weights: Float32Array): SsdMobilenetv1;

export declare function createTinyFaceDetector(weights: Float32Array): TinyFaceDetector;

export declare function createTinyYolov2(weights: Float32Array, withSeparableConvs?: boolean): TinyYolov2;

/**
 * Decoder parameters.
 */
export declare interface DecoderParams {
    conv1: FaceMeshConvBlockParams;
}

export declare type DefaultTinyYolov2NetParams = {
    conv0: ConvWithBatchNorm;
    conv1: ConvWithBatchNorm;
    conv2: ConvWithBatchNorm;
    conv3: ConvWithBatchNorm;
    conv4: ConvWithBatchNorm;
    conv5: ConvWithBatchNorm;
    conv6: ConvWithBatchNorm;
    conv7: ConvWithBatchNorm;
    conv8: ConvParams;
};

declare type DenseBlock3Params = {
    conv0: SeparableConvParams | ConvParams;
    conv1: SeparableConvParams;
    conv2: SeparableConvParams;
};

declare type DenseBlock4Params = DenseBlock3Params & {
    conv3: SeparableConvParams;
};

export declare class DetectAllFaceLandmarksTask<TSource extends WithFaceDetection<{}>> extends DetectFaceLandmarksTaskBase<WithFaceLandmarks<TSource>[], TSource[]> {
    run(): Promise<WithFaceLandmarks<TSource>[]>;
    withFaceExpressions(): PredictAllFaceExpressionsWithFaceAlignmentTask<WithFaceLandmarks<TSource>>;
    withAgeAndGender(): PredictAllAgeAndGenderWithFaceAlignmentTask<WithFaceLandmarks<TSource>>;
    withFaceDescriptors(): ComputeAllFaceDescriptorsTask<WithFaceLandmarks<TSource>>;
}

export declare function detectAllFaces(input: TNetInput, options?: FaceDetectionOptions): DetectAllFacesTask;

export declare class DetectAllFacesTask extends DetectFacesTaskBase<FaceDetection[]> {
    run(): Promise<FaceDetection[]>;
    private runAndExtendWithFaceDetections;
    withFaceLandmarks(useTinyLandmarkNet?: boolean): DetectAllFaceLandmarksTask<{
        detection: FaceDetection;
    }>;
    withFaceExpressions(): PredictAllFaceExpressionsTask<{
        detection: FaceDetection;
    }>;
    withAgeAndGender(): PredictAllAgeAndGenderTask<{
        detection: FaceDetection;
    }>;
}

/**
 * Detects the 68 point face landmark positions of the face shown in an image.
 *
 * @param inputs The face image extracted from the bounding box of a face. Can
 * also be an array of input images, which will be batch processed.
 * @returns 68 point face landmarks or array thereof in case of batch input.
 */
export declare const detectFaceLandmarks: (input: TNetInput) => Promise<FaceLandmarks68 | FaceLandmarks68[]>;

export declare class DetectFaceLandmarksTaskBase<TReturn, TParentReturn> extends ComposableTask<TReturn> {
    protected parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>;
    protected input: TNetInput;
    protected useTinyLandmarkNet: boolean;
    constructor(parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>, input: TNetInput, useTinyLandmarkNet: boolean);
    protected get landmarkNet(): FaceLandmark68Net | FaceLandmark68TinyNet;
}

/**
 * Detects the 68 point face landmark positions of the face shown in an image
 * using a tinier version of the 68 point face landmark model, which is slightly
 * faster at inference, but also slightly less accurate.
 *
 * @param inputs The face image extracted from the bounding box of a face. Can
 * also be an array of input images, which will be batch processed.
 * @returns 68 point face landmarks or array thereof in case of batch input.
 */
export declare const detectFaceLandmarksTiny: (input: TNetInput) => Promise<FaceLandmarks68 | FaceLandmarks68[]>;

export declare class DetectFacesTaskBase<TReturn> extends ComposableTask<TReturn> {
    protected input: TNetInput;
    protected options: FaceDetectionOptions;
    constructor(input: TNetInput, options?: FaceDetectionOptions);
}

export declare const detectLandmarks: (input: TNetInput) => Promise<FaceLandmarks68 | FaceLandmarks68[]>;

export declare function detectSingleFace(input: TNetInput, options?: FaceDetectionOptions): DetectSingleFaceTask;

export declare class DetectSingleFaceLandmarksTask<TSource extends WithFaceDetection<{}>> extends DetectFaceLandmarksTaskBase<WithFaceLandmarks<TSource> | undefined, TSource | undefined> {
    run(): Promise<WithFaceLandmarks<TSource> | undefined>;
    withFaceExpressions(): PredictSingleFaceExpressionsWithFaceAlignmentTask<WithFaceLandmarks<TSource>>;
    withAgeAndGender(): PredictSingleAgeAndGenderWithFaceAlignmentTask<WithFaceLandmarks<TSource>>;
    withFaceDescriptor(): ComputeSingleFaceDescriptorTask<WithFaceLandmarks<TSource>>;
}

export declare class DetectSingleFaceTask extends DetectFacesTaskBase<FaceDetection | undefined> {
    run(): Promise<FaceDetection | undefined>;
    private runAndExtendWithFaceDetection;
    withFaceLandmarks(useTinyLandmarkNet?: boolean): DetectSingleFaceLandmarksTask<{
        detection: FaceDetection;
    }>;
    withFaceExpressions(): PredictSingleFaceExpressionsTask<{
        detection: FaceDetection;
    }>;
    withAgeAndGender(): PredictSingleAgeAndGenderTask<{
        detection: FaceDetection;
    }>;
}

export declare class Dimensions implements IDimensions {
    private _width;
    private _height;
    constructor(width: number, height: number);
    get width(): number;
    get height(): number;
    reverse(): Dimensions;
}

/**
 * Configuration for document KYC verification.
 */
export declare interface DocumentKYCConfig {
    /** Base match threshold (default: 0.6) */
    baseMatchThreshold: number;
    /** Allow quality-based threshold adjustment (default: true) */
    adaptiveThreshold: boolean;
    /** Maximum threshold adjustment (default: 0.15) */
    maxThresholdAdjustment: number;
    /** Require manual review for medium confidence (default: true) */
    requireReviewForMedium: boolean;
    /** Document type being verified */
    documentType: DocumentType_2;
}

/**
 * Document types supported.
 */
declare type DocumentType_2 = 'rg' | 'cnh' | 'passport' | 'other';
export { DocumentType_2 as DocumentType }

declare namespace draw {
    export {
        drawContour,
        drawDetections,
        TDrawDetectionsInput,
        drawFaceExpressions,
        DrawFaceExpressionsInput,
        IDrawBoxOptions,
        DrawBoxOptions,
        DrawBox,
        drawFaceLandmarks,
        IDrawFaceLandmarksOptions,
        DrawFaceLandmarksOptions,
        DrawFaceLandmarks,
        DrawFaceLandmarksInput,
        AnchorPosition,
        IDrawTextFieldOptions,
        DrawTextFieldOptions,
        DrawTextField
    }
}
export { draw }

declare class DrawBox {
    box: Box;
    options: DrawBoxOptions;
    constructor(box: IBoundingBox | IRect, options?: IDrawBoxOptions);
    draw(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D): void;
}

declare class DrawBoxOptions {
    boxColor: string;
    lineWidth: number;
    drawLabelOptions: DrawTextFieldOptions;
    label?: string;
    constructor(options?: IDrawBoxOptions);
}

declare function drawContour(ctx: CanvasRenderingContext2D, points: Point[], isClosed?: boolean): void;

declare function drawDetections(canvasArg: string | HTMLCanvasElement, detections: TDrawDetectionsInput | Array<TDrawDetectionsInput>): void;

declare function drawFaceExpressions(canvasArg: string | HTMLCanvasElement, faceExpressions: DrawFaceExpressionsInput | Array<DrawFaceExpressionsInput>, minConfidence?: number, textFieldAnchor?: IPoint): void;

declare type DrawFaceExpressionsInput = FaceExpressions | WithFaceExpressions<{}>;

declare class DrawFaceLandmarks {
    faceLandmarks: FaceLandmarks;
    options: DrawFaceLandmarksOptions;
    constructor(faceLandmarks: FaceLandmarks, options?: IDrawFaceLandmarksOptions);
    draw(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D): void;
}

declare function drawFaceLandmarks(canvasArg: string | HTMLCanvasElement, faceLandmarks: DrawFaceLandmarksInput | Array<DrawFaceLandmarksInput>): void;

declare type DrawFaceLandmarksInput = FaceLandmarks | WithFaceLandmarks<WithFaceDetection<{}>>;

declare class DrawFaceLandmarksOptions {
    drawLines: boolean;
    drawPoints: boolean;
    lineWidth: number;
    pointSize: number;
    lineColor: string;
    pointColor: string;
    constructor(options?: IDrawFaceLandmarksOptions);
}

declare class DrawTextField {
    text: string[];
    anchor: IPoint;
    options: DrawTextFieldOptions;
    constructor(text: string | string[] | DrawTextField, anchor: IPoint, options?: IDrawTextFieldOptions);
    measureWidth(ctx: CanvasRenderingContext2D): number;
    measureHeight(): number;
    getUpperLeft(ctx: CanvasRenderingContext2D, canvasDims?: IDimensions): IPoint;
    draw(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D): void;
}

declare class DrawTextFieldOptions implements IDrawTextFieldOptions {
    anchorPosition: AnchorPosition;
    backgroundColor: string;
    fontColor: string;
    fontSize: number;
    fontStyle: string;
    padding: number;
    constructor(options?: IDrawTextFieldOptions);
}

/**
 * Encoder parameters.
 */
export declare interface EncoderParams {
    conv1: FaceMeshConvBlockParams;
    conv2: FaceMeshConvBlockParams;
    conv3: FaceMeshConvBlockParams;
    conv4: FaceMeshConvBlockParams;
    conv5: FaceMeshConvBlockParams;
}

export declare function ensureBackendInitialized(): Promise<BackendInitResult>;

export declare const env: {
    getEnv: typeof getEnv;
    setEnv: typeof setEnv;
    initialize: typeof initialize;
    createBrowserEnv: typeof createBrowserEnv;
    createFileSystem: typeof createFileSystem;
    createNodejsEnv: typeof createNodejsEnv;
    monkeyPatch: typeof monkeyPatch;
    isBrowser: typeof isBrowser;
    isNodejs: typeof isNodejs;
};

export declare type Environment = FileSystem_2 & {
    Canvas: typeof HTMLCanvasElement;
    CanvasRenderingContext2D: typeof CanvasRenderingContext2D;
    Image: typeof HTMLImageElement;
    ImageData: typeof ImageData;
    Video: typeof HTMLVideoElement;
    createCanvasElement: () => HTMLCanvasElement;
    createImageElement: () => HTMLImageElement;
    createVideoElement: () => HTMLVideoElement;
    fetch: (url: string, init?: RequestInit) => Promise<Response>;
};

export declare function euclideanDistance(arr1: number[] | Float32Array | null | undefined, arr2: number[] | Float32Array | null | undefined): number;

export declare function extendWithAge<TSource>(sourceObj: TSource, age: number): WithAge<TSource>;

export declare function extendWithFaceDescriptor<TSource>(sourceObj: TSource, descriptor: Float32Array): WithFaceDescriptor<TSource>;

export declare function extendWithFaceDetection<TSource>(sourceObj: TSource, detection: FaceDetection): WithFaceDetection<TSource>;

export declare function extendWithFaceExpressions<TSource>(sourceObj: TSource, expressions: FaceExpressions): WithFaceExpressions<TSource>;

export declare function extendWithFaceLandmarks<TSource extends WithFaceDetection<{}>, TFaceLandmarks extends FaceLandmarks = FaceLandmarks68>(sourceObj: TSource, unshiftedLandmarks: TFaceLandmarks): WithFaceLandmarks<TSource, TFaceLandmarks>;

export declare function extendWithGender<TSource>(sourceObj: TSource, gender: Gender, genderProbability: number): WithGender<TSource>;

/**
 * Extracts the image regions containing the detected faces.
 *
 * @param input The image that face detection has been performed on.
 * @param detections The face detection results or face bounding boxes for that image.
 * @returns The Canvases of the corresponding image region for each detected face.
 */
export declare function extractFaces(input: TNetInput, detections: Array<FaceDetection | Rect>): Promise<HTMLCanvasElement[]>;

/**
 * Extracts the tensors of the image regions containing the detected faces.
 * Useful if you want to compute the face descriptors for the face images.
 * Using this method is faster then extracting a canvas for each face and
 * converting them to tensors individually.
 *
 * @param imageTensor The image tensor that face detection has been performed on.
 * @param detections The face detection results or face bounding boxes for that image.
 * @returns Tensors of the corresponding image region for each detected face.
 */
export declare function extractFaceTensors(imageTensor: tf.Tensor3D | tf.Tensor4D, detections: Array<FaceDetection | Rect>): Promise<tf.Tensor3D[]>;

export declare const FACE_EXPRESSION_LABELS: readonly ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];

export declare class FaceDetection extends ObjectDetection implements IFaceDetection {
    constructor(score: number, relativeBox: Rect, imageDims: IDimensions);
    forSize(width: number, height: number): FaceDetection;
}

export declare type FaceDetectionFunction = (input: TNetInput) => Promise<FaceDetection[]>;

export declare class FaceDetectionNet extends SsdMobilenetv1 {
}

export declare type FaceDetectionOptions = TinyFaceDetectorOptions | SsdMobilenetv1Options | TinyYolov2Options;

export declare class FaceExpressionNet extends FaceProcessor<FaceFeatureExtractorParams> {
    constructor(faceFeatureExtractor?: FaceFeatureExtractor);
    forwardInput(input: NetInput | tf.Tensor4D): tf.Tensor2D;
    forward(input: TNetInput): Promise<tf.Tensor2D>;
    predictExpressions(input: TNetInput): Promise<FaceExpressions | FaceExpressions[] | undefined>;
    protected getDefaultModelName(): string;
    protected getClassifierChannelsIn(): number;
    protected getClassifierChannelsOut(): number;
}

export declare class FaceExpressions {
    neutral: number;
    happy: number;
    sad: number;
    angry: number;
    fearful: number;
    disgusted: number;
    surprised: number;
    constructor(probabilities: number[] | Float32Array);
    asSortedArray(): {
        expression: "neutral" | "happy" | "sad" | "angry" | "fearful" | "disgusted" | "surprised";
        probability: number;
    }[];
}

declare class FaceFeatureExtractor extends NeuralNetwork<FaceFeatureExtractorParams> implements IFaceFeatureExtractor<FaceFeatureExtractorParams> {
    constructor();
    forwardInput(input: NetInput): tf.Tensor4D;
    forward(input: TNetInput): Promise<tf.Tensor4D>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: FaceFeatureExtractorParams;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: FaceFeatureExtractorParams;
        paramMappings: ParamMapping[];
    };
}

declare type FaceFeatureExtractorParams = {
    dense0: DenseBlock4Params;
    dense1: DenseBlock4Params;
    dense2: DenseBlock4Params;
    dense3: DenseBlock4Params;
};

export declare class FaceLandmark68Net extends FaceLandmark68NetBase<FaceFeatureExtractorParams> {
    constructor(faceFeatureExtractor?: FaceFeatureExtractor);
    protected getDefaultModelName(): string;
    protected getClassifierChannelsIn(): number;
}

declare abstract class FaceLandmark68NetBase<TExtractorParams extends FaceFeatureExtractorParams | TinyFaceFeatureExtractorParams> extends FaceProcessor<TExtractorParams> {
    postProcess(output: tf.Tensor2D, inputSize: number, originalDimensions: IDimensions[]): tf.Tensor2D;
    forwardInput(input: NetInput): tf.Tensor2D;
    forward(input: TNetInput): Promise<tf.Tensor2D>;
    detectLandmarks(input: TNetInput): Promise<FaceLandmarks68 | FaceLandmarks68[]>;
    protected getClassifierChannelsOut(): number;
}

export declare class FaceLandmark68TinyNet extends FaceLandmark68NetBase<TinyFaceFeatureExtractorParams> {
    constructor(faceFeatureExtractor?: TinyFaceFeatureExtractor);
    protected getDefaultModelName(): string;
    protected getClassifierChannelsIn(): number;
}

export declare class FaceLandmarkNet extends FaceLandmark68Net {
}

export declare class FaceLandmarks implements IFaceLandmarks {
    protected _shift: Point;
    protected _positions: Point[];
    protected _imgDims: Dimensions;
    constructor(relativeFaceLandmarkPositions: Point[], imgDims: IDimensions, shift?: Point);
    get shift(): Point;
    get imageWidth(): number;
    get imageHeight(): number;
    get positions(): Point[];
    get relativePositions(): Point[];
    forSize<T extends FaceLandmarks>(width: number, height: number): T;
    shiftBy<T extends FaceLandmarks>(x: number, y: number): T;
    shiftByPoint<T extends FaceLandmarks>(pt: Point): T;
    /**
     * Aligns the face landmarks after face detection from the relative positions of the faces
     * bounding box, or it's current shift. This function should be used to align the face images
     * after face detection has been performed, before they are passed to the face recognition net.
     * This will make the computed face descriptor more accurate.
     *
     * @param detection (optional) The bounding box of the face or the face detection result. If
     * no argument was passed the position of the face landmarks are assumed to be relative to
     * it's current shift.
     * @returns The bounding box of the aligned face.
     */
    align(detection?: FaceDetection | IRect | IBoundingBox | null, options?: {
        useDlibAlignment?: boolean;
        minBoxPadding?: number;
    }): Box;
    private alignDlib;
    private alignMinBbox;
    protected getRefPointsForAlignment(): Point[];
}

export declare class FaceLandmarks5 extends FaceLandmarks {
    protected getRefPointsForAlignment(): Point[];
}

export declare class FaceLandmarks68 extends FaceLandmarks {
    getJawOutline(): Point[];
    getLeftEyeBrow(): Point[];
    getRightEyeBrow(): Point[];
    getNose(): Point[];
    getLeftEye(): Point[];
    getRightEye(): Point[];
    getMouth(): Point[];
    protected getRefPointsForAlignment(): Point[];
}

export declare class FaceMatch implements IFaceMatch {
    private _label;
    private _distance;
    constructor(label: string, distance: number);
    get label(): string;
    get distance(): number;
    toString(withDistance?: boolean): string;
}

export declare class FaceMatcher {
    private _labeledDescriptors;
    private _distanceThreshold;
    constructor(inputs: LabeledFaceDescriptors | WithFaceDescriptor<any> | Float32Array | Array<LabeledFaceDescriptors | WithFaceDescriptor<any> | Float32Array>, distanceThreshold?: number);
    get labeledDescriptors(): LabeledFaceDescriptors[];
    get distanceThreshold(): number;
    computeMeanDistance(queryDescriptor: Float32Array, descriptors: Float32Array[]): number;
    matchDescriptor(queryDescriptor: Float32Array): FaceMatch;
    findBestMatch(queryDescriptor: Float32Array): FaceMatch;
    toJSON(): any;
    static fromJSON(json: any): FaceMatcher;
}

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
export declare class FaceMesh extends NeuralNetwork<NetParams_7> {
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
        params: NetParams_7;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams_7;
        paramMappings: ParamMapping[];
    };
}

/**
 * Number of landmarks in different configurations.
 */
export declare const FACEMESH_LANDMARK_COUNTS: {
    /**
     * Base FaceMesh landmarks.
     */
    readonly BASE: 468;
    /**
     * With refinement around lips and eyes.
     */
    readonly REFINED: 478;
    /**
     * Legacy 68-point landmarks.
     */
    readonly LEGACY_68: 68;
    /**
     * Legacy 5-point landmarks.
     */
    readonly LEGACY_5: 5;
};

/**
 * Semantic region indices for FaceMesh.
 */
export declare const FACEMESH_REGIONS: {
    readonly FACE_OVAL: readonly [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
    readonly LEFT_EYE: readonly [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    readonly RIGHT_EYE: readonly [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
    readonly LEFT_EYEBROW: readonly [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
    readonly RIGHT_EYEBROW: readonly [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
    readonly NOSE: readonly [1, 2, 98, 327, 4, 5, 6, 168, 195, 197, 419, 351, 412, 343, 437, 420, 456, 248, 281, 275, 274, 354, 370, 94, 19];
    readonly LIPS_OUTER: readonly [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    readonly LIPS_INNER: readonly [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
    readonly LEFT_IRIS: readonly [468, 469, 470, 471, 472];
    readonly RIGHT_IRIS: readonly [473, 474, 475, 476, 477];
};

/**
 * Complete mapping from FaceMesh indices to 68-point format.
 * Each index in this array corresponds to a 68-point landmark,
 * and the value is the corresponding FaceMesh index.
 */
export declare const FACEMESH_TO_68_MAPPING: number[];

/**
 * FaceMesh convolution block parameters.
 */
export declare interface FaceMeshConvBlockParams {
    weights: tf.Tensor4D;
    bn_mean: tf.Tensor1D;
    bn_variance: tf.Tensor1D;
    bn_offset: tf.Tensor1D;
    bn_scale: tf.Tensor1D;
}

/**
 * FaceMesh 468/478 point landmarks with 3D coordinates.
 */
export declare class FaceMeshLandmarks extends FaceLandmarks {
    private _zValues;
    constructor(positions: Point[], imageDims: IDimensions, zValues: number[], shift?: Point);
    /**
     * Get Z-coordinates for all landmarks.
     */
    get zValues(): number[];
    /**
     * Get 3D position for a specific landmark.
     */
    getPosition3D(idx: number): {
        x: number;
        y: number;
        z: number;
    };
    /**
     * Get all 3D positions.
     */
    get positions3D(): Array<{
        x: number;
        y: number;
        z: number;
    }>;
    /**
     * Helper to filter undefined values from Point arrays.
     */
    private filterPoints;
    /**
     * Get face oval landmarks (36 points).
     */
    getFaceOval(): Point[];
    /**
     * Get left eye landmarks.
     */
    getLeftEye(): Point[];
    /**
     * Get right eye landmarks.
     */
    getRightEye(): Point[];
    /**
     * Get left eyebrow landmarks.
     */
    getLeftEyebrow(): Point[];
    /**
     * Get right eyebrow landmarks.
     */
    getRightEyebrow(): Point[];
    /**
     * Get nose landmarks.
     */
    getNose(): Point[];
    /**
     * Get lips landmarks (inner + outer).
     */
    getLips(): Point[];
    /**
     * Get left iris landmarks (if refined).
     */
    getLeftIris(): Point[];
    /**
     * Get right iris landmarks (if refined).
     */
    getRightIris(): Point[];
    /**
     * Convert to 68-point FaceLandmarks68 for compatibility.
     */
    toLandmarks68(): FaceLandmarks68;
    /**
     * Estimate head pose from landmarks.
     * Returns rotation angles in degrees.
     */
    estimateHeadPose(): {
        pitch: number;
        yaw: number;
        roll: number;
    };
}

export declare class FaceMeshOptions {
    readonly maxFaces: number;
    readonly refineLandmarks: boolean;
    readonly minDetectionConfidence: number;
    readonly minTrackingConfidence: number;
    readonly enableFallback: boolean;
    constructor(options?: IFaceMeshOptions);
}

declare abstract class FaceProcessor<TExtractorParams extends FaceFeatureExtractorParams | TinyFaceFeatureExtractorParams> extends NeuralNetwork<NetParams_2> {
    protected _faceFeatureExtractor: IFaceFeatureExtractor<TExtractorParams>;
    constructor(_name: string, faceFeatureExtractor: IFaceFeatureExtractor<TExtractorParams>);
    get faceFeatureExtractor(): IFaceFeatureExtractor<TExtractorParams>;
    protected abstract getDefaultModelName(): string;
    protected abstract getClassifierChannelsIn(): number;
    protected abstract getClassifierChannelsOut(): number;
    runNet(input: NetInput | tf.Tensor4D): tf.Tensor2D;
    dispose(throwOnRedispose?: boolean): void;
    loadClassifierParams(weights: Float32Array): void;
    extractClassifierParams(weights: Float32Array): {
        params: NetParams_2;
        paramMappings: ParamMapping[];
    };
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: NetParams_2;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams_2;
        paramMappings: ParamMapping[];
    };
}

/**
 * Face quality metrics for KYC validation.
 */
export declare interface FaceQualityMetrics {
    /** Overall quality score (0-1) */
    overallScore: number;
    /** Sharpness/blur score (0-1, higher = sharper) */
    sharpness: number;
    /** Brightness score (0-1, 0.5 = optimal) */
    brightness: number;
    /** Contrast score (0-1) */
    contrast: number;
    /** Face size relative to image (0-1) */
    faceSize: number;
    /** Head pose angles in degrees */
    pose: {
        yaw: number;
        pitch: number;
        roll: number;
    };
    /** Is the pose within acceptable range? */
    isFrontal: boolean;
    /** Are both eyes visible? */
    eyesVisible: boolean;
    /** Confidence of the detection */
    detectionConfidence: number;
}

export declare class FaceRecognitionNet extends NeuralNetwork<NetParams_3> {
    constructor();
    forwardInput(input: NetInput): tf.Tensor2D;
    forward(input: TNetInput): Promise<tf.Tensor2D>;
    computeFaceDescriptor(input: TNetInput): Promise<Float32Array | Float32Array[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: NetParams_3;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams_3;
        paramMappings: ParamMapping[];
    };
}

/**
 * Fully connected layer parameters.
 */
export declare interface FCParams {
    weights: tf.Tensor2D;
    bias: tf.Tensor1D;
}

declare type FCParams_2 = {
    weights: tf.Tensor2D;
    bias: tf.Tensor1D;
};

export declare function fetchImage(uri: string): Promise<HTMLImageElement>;

export declare function fetchJson<T>(uri: string): Promise<T>;

export declare function fetchNetWeights(uri: string): Promise<Float32Array>;

export declare function fetchOrThrow(url: string, init?: RequestInit): Promise<Response>;

export declare function fetchVideo(uri: string): Promise<HTMLVideoElement>;

declare type FileSystem_2 = {
    readFile: (filePath: string) => Promise<string | Buffer>;
};
export { FileSystem_2 as FileSystem }

export declare enum Gender {
    FEMALE = "female",
    MALE = "male"
}

/**
 * Get information about all available backends.
 */
export declare function getAvailableBackends(): {
    backend: BackendType;
    supported: boolean;
}[];

/**
 * Get performance characteristics for each backend.
 */
export declare function getBackendPerformance(): BackendPerformance[];

declare function getCenterPoint(pts: Point[]): Point;

export declare function getContext2dOrThrow(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D): CanvasRenderingContext2D;

/**
 * Get current TensorFlow.js backend name.
 */
export declare function getCurrentBackend(): string;

declare function getEnv(): Environment;

export declare function getMediaDimensions(input: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | IDimensions): Dimensions;

/**
 * Get detailed WebGPU support information.
 * Returns adapter and device info if available.
 */
export declare function getWebGPUInfo(): Promise<WebGPUInfo>;

/**
 * BlazeFace detection head parameters.
 */
export declare interface HeadParams {
    weights: tf.Tensor4D;
    bias: tf.Tensor1D;
}

/**
 * Options for AdaFace face recognition.
 */
export declare interface IAdaFaceOptions {
    /**
     * Input size for face crops. AdaFace uses 112x112.
     * Default: 112
     */
    inputSize?: number;
    /**
     * Descriptor size. AdaFace typically uses 512-d embeddings.
     * Default: 512
     */
    descriptorSize?: 128 | 256 | 512;
    /**
     * If true, falls back to FaceNet when AdaFace fails.
     * Default: true
     */
    enableFallback?: boolean;
    /**
     * Quality threshold for adaptive margin.
     * AdaFace adjusts margin based on image quality.
     * Default: 0.5
     */
    qualityThreshold?: number;
    /**
     * If true and both models are loaded, blend AdaFace and FaceNet descriptors.
     * This can improve accuracy for challenging images.
     * Default: false
     */
    blendDescriptors?: boolean;
    /**
     * Weight for AdaFace descriptor when blending (0-1).
     * FaceNet weight = 1 - blendWeight.
     * Default: 0.7 (AdaFace gets 70%, FaceNet 30%)
     */
    blendWeight?: number;
}

/**
 * Options for BlazeFace detector.
 */
export declare interface IBlazeFaceOptions {
    /**
     * Minimum confidence threshold for detections (0-1).
     * Default: 0.5
     */
    minConfidence?: number;
    /**
     * Maximum number of faces to detect.
     * Default: 10
     */
    maxResults?: number;
    /**
     * Input size for the model. BlazeFace supports 128 or 256.
     * Larger size = more accurate but slower.
     * Default: 128
     */
    inputSize?: 128 | 256;
    /**
     * IOU threshold for non-max suppression.
     * Default: 0.3
     */
    iouThreshold?: number;
    /**
     * If true, falls back to SSD MobileNetv1 when BlazeFace fails.
     * Default: true
     */
    enableFallback?: boolean;
}

export declare interface IBoundingBox {
    left: number;
    top: number;
    right: number;
    bottom: number;
}

export declare interface IDimensions {
    width: number;
    height: number;
}

declare interface IDrawBoxOptions {
    boxColor?: string;
    lineWidth?: number;
    drawLabelOptions?: IDrawTextFieldOptions;
    label?: string;
}

declare interface IDrawFaceLandmarksOptions {
    drawLines?: boolean;
    drawPoints?: boolean;
    lineWidth?: number;
    pointSize?: number;
    lineColor?: string;
    pointColor?: string;
}

declare interface IDrawTextFieldOptions {
    anchorPosition?: AnchorPosition;
    backgroundColor?: string;
    fontColor?: string;
    fontSize?: number;
    fontStyle?: string;
    padding?: number;
}

export declare interface IFaceDetection {
    score: number;
    box: Box;
}

declare interface IFaceFeatureExtractor<TNetParams extends TinyFaceFeatureExtractorParams | FaceFeatureExtractorParams> extends NeuralNetwork<TNetParams> {
    forwardInput(input: NetInput): tf.Tensor4D;
    forward(input: TNetInput): Promise<tf.Tensor4D>;
}

export declare interface IFaceLandmarks {
    positions: Point[];
    shift: Point;
}

export declare interface IFaceMatch {
    label: string;
    distance: number;
}

/**
 * Options for FaceMesh landmark detector.
 */
export declare interface IFaceMeshOptions {
    /**
     * Maximum number of faces to detect landmarks for.
     * Default: 1
     */
    maxFaces?: number;
    /**
     * Whether to refine landmarks around lips and eyes.
     * Adds ~80 additional landmarks for these regions.
     * Default: false
     */
    refineLandmarks?: boolean;
    /**
     * Minimum detection confidence (0-1).
     * Default: 0.5
     */
    minDetectionConfidence?: number;
    /**
     * Minimum tracking confidence (0-1).
     * Default: 0.5
     */
    minTrackingConfidence?: number;
    /**
     * If true, falls back to 68-point landmarks when FaceMesh fails.
     * Default: true
     */
    enableFallback?: boolean;
}

/**
 * Image Quality Analyzer using real pixel analysis.
 *
 * Implements:
 * - Laplacian variance for sharpness/blur detection
 * - Mean pixel intensity for brightness
 * - Standard deviation for contrast
 */
export declare class ImageQualityAnalyzer {
    /**
     * Analyze image quality for the face region.
     */
    analyze(input: TNetInput, detection: FaceDetection): Promise<ImageQualityResult>;
    /**
     * Convert RGB tensor to grayscale.
     */
    private toGrayscale;
    /**
     * Extract face region from grayscale image.
     */
    private extractFaceRegion;
    /**
     * Calculate sharpness using Laplacian variance.
     * Higher variance = sharper image.
     */
    private calculateSharpness;
    /**
     * Calculate brightness as mean pixel intensity.
     */
    private calculateBrightness;
    /**
     * Calculate contrast as standard deviation of pixel intensities.
     */
    private calculateContrast;
    /**
     * Get default result when analysis fails.
     */
    private getDefaultResult;
}

/**
 * Singleton instance for convenience.
 */
export declare const imageQualityAnalyzer: ImageQualityAnalyzer;

/**
 * Image quality metrics result.
 */
export declare interface ImageQualityResult {
    /** Sharpness score (0-1, higher = sharper) */
    sharpness: number;
    /** Brightness score (0-1, 0.5 = optimal) */
    brightness: number;
    /** Contrast score (0-1, higher = more contrast) */
    contrast: number;
    /** Whether the face region was successfully extracted */
    faceRegionExtracted: boolean;
}

/**
 * Source type for the image being analyzed.
 */
export declare type ImageSourceType = 'selfie' | 'document' | 'unknown';

export declare function imageTensorToCanvas(imgTensor: tf.Tensor, canvas?: HTMLCanvasElement): Promise<HTMLCanvasElement>;

export declare function imageToSquare(input: HTMLImageElement | HTMLCanvasElement, inputSize: number, centerImage?: boolean): HTMLCanvasElement;

/**
 * Recommended way to initialize face-api.
 * Automatically selects the best backend with WebGPU priority.
 *
 * @example
 * ```typescript
 * import * as faceapi from '@vladmandic/face-api';
 *
 * // Initialize with best available backend
 * await faceapi.init();
 *
 * // Or with specific preference
 * await faceapi.init({ preferredBackend: 'webgl' });
 * ```
 */
export declare function init(options?: {
    preferredBackend?: BackendType;
    silent?: boolean;
}): Promise<BackendInitResult>;

export declare function initBestBackend(preferredBackend?: BackendType): Promise<BackendInitResult>;

declare function initialize(): void | null;

/**
 * Initialize WebGPU backend for TensorFlow.js.
 * This must be called before using any face-api.js functions.
 *
 * @throws Error if WebGPU is not supported
 * @returns Promise that resolves when backend is ready
 *
 * @example
 * ```typescript
 * import * as faceapi from '@vladmandic/face-api/webgpu';
 * import { initWebGPU } from '@vladmandic/face-api/webgpu';
 *
 * await initWebGPU();
 * // Now use face-api normally
 * const detections = await faceapi.detectAllFaces(image);
 * ```
 */
export declare function initWebGPU(): Promise<void>;

/**
 * Initialize WebGPU with automatic fallback to WebGL/CPU.
 * Unlike initWebGPU(), this never throws and always succeeds.
 *
 * @returns Result indicating which backend was initialized
 *
 * @example
 * ```typescript
 * const { backend, isPreferred } = await initWebGPUWithFallback();
 * if (backend === 'webgpu') {
 *   console.log('🚀 WebGPU enabled - maximum performance!');
 * } else {
 *   console.log(`Using ${backend} fallback`);
 * }
 * ```
 */
export declare function initWebGPUWithFallback(): Promise<BackendInitResult>;

export declare function inverseSigmoid(x: number): number;

export declare function iou(box1: Box, box2: Box, isIOU?: boolean): number;

export declare interface IPoint {
    x: number;
    y: number;
}

export declare interface IRect {
    x: number;
    y: number;
    width: number;
    height: number;
}

/**
 * IR-SE block parameters.
 */
export declare interface IRSEBlockParams {
    depthwise: tf.Tensor4D;
    bn1_mean: tf.Tensor1D;
    bn1_variance: tf.Tensor1D;
    bn1_offset: tf.Tensor1D;
    bn1_scale: tf.Tensor1D;
    se: SEParams;
    pointwise: tf.Tensor4D;
    bn2_mean: tf.Tensor1D;
    bn2_variance: tf.Tensor1D;
    bn2_offset: tf.Tensor1D;
    bn2_scale: tf.Tensor1D;
}

declare function isBrowser(): boolean;

declare function isDimensions(obj: any): boolean;

declare function isEven(num: number): boolean;

declare function isFloat(num: number): boolean;

export declare function isMediaElement(input: any): input is HTMLCanvasElement | HTMLImageElement | HTMLVideoElement;

export declare function isMediaLoaded(media: HTMLImageElement | HTMLVideoElement): boolean;

declare function isNodejs(): boolean;

export declare interface ISsdMobilenetv1Options {
    minConfidence?: number;
    maxResults?: number;
}

declare function isTensor(tensor: any, dim: number): boolean;

declare function isTensor1D(tensor: any): tensor is tf.Tensor1D;

declare function isTensor2D(tensor: any): tensor is tf.Tensor2D;

declare function isTensor3D(tensor: any): tensor is tf.Tensor3D;

declare function isTensor4D(tensor: any): tensor is tf.Tensor4D;

declare function isValidNumber(num: any): boolean;

declare function isValidProbablitiy(num: any): boolean;

/**
 * Initialize the best available backend with automatic fallback.
 * Tries WebGPU first, falls back to WebGL, then CPU.
 *
 * @param preferredBackend - Preferred backend to try first (default: 'webgpu')
 * @returns Result indicating which backend was initialized
 *
 * @example
 * ```typescript
 * const result = await initBestBackend();
 * console.log(`Using ${result.backend} backend`);
 * if (!result.isPreferred) {
 *   console.log(`Fallback reason: ${result.fallbackReason}`);
 * }
 * ```
 */
/**
 * Check if WASM backend is available.
 */
export declare function isWasmSupported(): boolean;

/**
 * Check if WebGL is supported.
 */
export declare function isWebGLSupported(): boolean;

/**
 * Check if WebGPU backend is currently active.
 */
export declare function isWebGPUActive(): boolean;

/**
 * Check if WebGPU is supported in the current environment.
 * WebGPU requires Chrome 113+, Edge 113+, or compatible browsers.
 */
export declare function isWebGPUSupported(): boolean;

export declare function isWithAge(obj: any): obj is WithAge<{}>;

export declare function isWithFaceDetection(obj: any): obj is WithFaceDetection<{}>;

export declare function isWithFaceExpressions(obj: any): obj is WithFaceExpressions<{}>;

export declare function isWithFaceLandmarks(obj: any): obj is WithFaceLandmarks<WithFaceDetection<{}>, FaceLandmarks>;

export declare function isWithGender(obj: any): obj is WithGender<{}>;

export declare type ITinyFaceDetectorOptions = ITinyYolov2Options;

export declare interface ITinyYolov2Options {
    inputSize?: number;
    scoreThreshold?: number;
}

/**
 * KYC analysis result.
 */
export declare interface KYCAnalysisResult {
    /** Was a valid face detected? */
    faceDetected: boolean;
    /** Detection details */
    detection: FaceDetection | null;
    /** Facial landmarks */
    landmarks: FaceLandmarks68 | null;
    /** Face descriptor for matching */
    descriptor: Float32Array | null;
    /** Quality metrics */
    quality: FaceQualityMetrics | null;
    /** Validation issues found */
    issues: string[];
    /** Is the image acceptable for KYC? */
    isAcceptable: boolean;
    /** Recommendations for improvement */
    recommendations: string[];
}

/**
 * Configuration for KYC analysis.
 */
export declare interface KYCConfig {
    /** Minimum detection confidence (default: 0.8) */
    minDetectionConfidence: number;
    /** Minimum face size as fraction of image (default: 0.1) */
    minFaceSize: number;
    /** Maximum face size as fraction of image (default: 0.9) */
    maxFaceSize: number;
    /** Maximum yaw angle in degrees (default: 15) */
    maxYaw: number;
    /** Maximum pitch angle in degrees (default: 15) */
    maxPitch: number;
    /** Maximum roll angle in degrees (default: 10) */
    maxRoll: number;
    /** Minimum sharpness score (default: 0.3) */
    minSharpness: number;
    /** Match threshold for face comparison (default: 0.6) */
    matchThreshold: number;
}

/**
 * KYC Document Analyzer - Specialized for ID document vs selfie verification.
 *
 * Uses state-of-the-art models:
 * - **BlazeFace**: Ultra-fast detector (~98% accuracy, 200-1000+ FPS)
 * - **AdaFace**: Adaptive face recognition (99.82%+ LFW accuracy)
 *
 * Handles real-world challenges:
 * - Document photos are often old, low quality, printed/scanned
 * - Selfies are current, higher quality, but may have different lighting
 * - Age difference between document and current appearance
 * - Different thresholds for document vs selfie quality
 * - Adaptive matching based on image quality
 *
 * @example
 * ```typescript
 * const analyzer = new KYCDocumentAnalyzer();
 * await analyzer.load('/models');
 *
 * const result = await analyzer.verifyIdentity(
 *   selfieImage,
 *   documentImage,
 *   { documentType: 'cnh' }
 * );
 *
 * if (result.status === 'approved') {
 *   console.log(`✅ Aprovado - Similaridade: ${result.match.similarityPercent}%`);
 * } else if (result.status === 'needs_review') {
 *   console.log('⚠️ Necessita revisão manual');
 *   console.log('Notas:', result.operatorNotes);
 * } else {
 *   console.log('❌ Rejeitado');
 *   console.log('Motivos:', result.reasons);
 * }
 * ```
 */
export declare class KYCDocumentAnalyzer {
    private detector;
    private tinyDetector;
    private landmarkNet;
    private recognitionNet;
    private qualityAnalyzer;
    private config;
    private _isLoaded;
    private _officialBlazeFace;
    constructor(config?: Partial<DocumentKYCConfig>);
    /**
     * Check if official BlazeFace is loaded.
     */
    get isOfficialBlazeFaceLoaded(): boolean;
    /**
     * Load all required models.
     * Also loads the official BlazeFace model from TensorFlow Hub for better small face detection.
     */
    load(modelPath: string): Promise<void>;
    /**
     * Check if models are loaded.
     */
    get isLoaded(): boolean;
    /**
     * Analyze a selfie image.
     */
    analyzeSelfie(input: TNetInput): Promise<SourceAnalysisResult>;
    /**
     * Analyze a document photo.
     */
    analyzeDocument(input: TNetInput): Promise<SourceAnalysisResult>;
    /**
     * Complete identity verification: selfie vs document.
     */
    verifyIdentity(selfie: TNetInput, document: TNetInput, options?: Partial<DocumentKYCConfig>): Promise<KYCVerificationResult>;
    /**
     * Analyze an image with source-specific thresholds.
     * Uses BlazeFace for detection and AdaFace for recognition.
     */
    private analyzeImage;
    /**
     * Detect faces with image upscaling (for very small faces in documents).
     * Upscales the image 2x-3x to make small faces more detectable.
     */
    private detectWithUpscale;
    /**
     * Detect faces using the official BlazeFace model from TensorFlow Hub.
     * This model is optimized for detecting faces of various sizes.
     */
    private detectWithOfficialBlazeFace;
    /**
     * Calculate quality metrics for an image.
     * Uses real pixel analysis with Laplacian variance for sharpness.
     * Leverages BlazeFace keypoints for more accurate pose estimation when available.
     */
    private calculateQuality;
    /**
     * Estimate pose from BlazeFace keypoints (6-point).
     * More accurate than using 68-point landmarks for pose estimation.
     */
    private estimatePoseFromKeypoints;
    /**
     * Estimate pose from landmarks.
     */
    private estimatePose;
    /**
     * Calculate overall quality score.
     */
    private calculateQualityScore;
    /**
     * Identify issues with the image.
     */
    private identifyIssues;
    /**
     * Check if image meets acceptability criteria.
     */
    private checkAcceptability;
    /**
     * Calculate adaptive threshold based on image quality.
     */
    private calculateAdaptiveThreshold;
    /**
     * Determine confidence level.
     */
    private determineConfidence;
    /**
     * Calculate risk score (0-100).
     */
    private calculateRiskScore;
    /**
     * Translate confidence level to Portuguese.
     */
    private translateConfidence;
    /**
     * Create empty result for failed detection.
     */
    private createEmptyResult;
    /**
     * Create rejection result.
     */
    private createRejectionResult;
    /**
     * Dispose of resources.
     */
    dispose(): void;
}

/**
 * KYC Face Analyzer - Optimized for identity verification.
 *
 * Features:
 * - High-precision face detection
 * - Face quality assessment (blur, lighting, pose)
 * - Face matching between ID and selfie
 * - Detailed validation feedback
 *
 * @example
 * ```typescript
 * const analyzer = new KYCFaceAnalyzer();
 * await analyzer.load('/models');
 *
 * // Analyze ID document photo
 * const idResult = await analyzer.analyzeForKYC(idPhoto);
 * if (!idResult.isAcceptable) {
 *   console.log('Issues:', idResult.issues);
 *   console.log('Recommendations:', idResult.recommendations);
 * }
 *
 * // Compare ID with selfie
 * const matchResult = await analyzer.compareFaces(idPhoto, selfiePhoto);
 * if (matchResult.isMatch) {
 *   console.log(`Match confidence: ${matchResult.confidence}`);
 * }
 * ```
 */
export declare class KYCFaceAnalyzer {
    private detector;
    private landmarkNet;
    private recognitionNet;
    private qualityAnalyzer;
    private config;
    private _isLoaded;
    constructor(config?: Partial<KYCConfig>);
    /**
     * Load all required models.
     */
    load(modelPath: string): Promise<void>;
    /**
     * Check if models are loaded.
     */
    get isLoaded(): boolean;
    /**
     * Analyze a face image for KYC compliance.
     */
    analyzeForKYC(input: TNetInput): Promise<KYCAnalysisResult>;
    /**
     * Compare two face images (e.g., ID photo vs selfie).
     */
    compareFaces(idPhoto: TNetInput, selfie: TNetInput): Promise<KYCMatchResult>;
    /**
     * Calculate face quality metrics.
     */
    private calculateQuality;
    /**
     * Estimate head pose from landmarks.
     */
    private estimatePose;
    /**
     * Calculate image quality metrics from the face region.
     * Uses real pixel analysis with Laplacian variance for sharpness,
     * mean intensity for brightness, and standard deviation for contrast.
     */
    private calculateImageQuality;
    /**
     * Calculate overall quality score.
     */
    private calculateOverallScore;
    /**
     * Validate quality metrics and add issues/recommendations.
     */
    private validateQuality;
    /**
     * Dispose of resources.
     */
    dispose(): void;
}

/**
 * Result of comparing two faces for KYC verification.
 */
export declare interface KYCMatchResult {
    /** Are the faces from the same person? */
    isMatch: boolean;
    /** Similarity score (0-1, higher = more similar) */
    similarity: number;
    /** Euclidean distance between descriptors */
    distance: number;
    /** Confidence level of the match */
    confidence: 'high' | 'medium' | 'low';
    /** Quality of ID photo */
    idQuality: FaceQualityMetrics | null;
    /** Quality of selfie */
    selfieQuality: FaceQualityMetrics | null;
}

/**
 * Complete KYC verification result.
 */
export declare interface KYCVerificationResult {
    /** Selfie analysis */
    selfie: SourceAnalysisResult;
    /** Document analysis */
    document: SourceAnalysisResult;
    /** Match result */
    match: {
        /** Final decision */
        isMatch: boolean;
        /** Raw distance between descriptors */
        distance: number;
        /** Similarity percentage (0-100) */
        similarityPercent: number;
        /** Adjusted threshold used (varies by quality) */
        thresholdUsed: number;
        /** Confidence level */
        confidence: 'very_high' | 'high' | 'medium' | 'low' | 'very_low';
        /** Risk score (0-100, lower = safer) */
        riskScore: number;
    };
    /** Overall verification status */
    status: 'approved' | 'needs_review' | 'rejected';
    /** Reasons for the decision */
    reasons: string[];
    /** Recommendations for the operator */
    operatorNotes: string[];
}

export declare class LabeledBox extends Box {
    static assertIsValidLabeledBox(box: any, callee: string): void;
    private _label;
    constructor(box: IBoundingBox | IRect | any, label: number);
    get label(): number;
}

export declare class LabeledFaceDescriptors {
    private _label;
    private _descriptors;
    constructor(label: string, descriptors: Float32Array[]);
    get label(): string;
    get descriptors(): Float32Array[];
    toJSON(): any;
    static fromJSON(json: any): LabeledFaceDescriptors;
}

/**
 * Landmark head parameters.
 */
export declare interface LandmarkHeadParams {
    weights: tf.Tensor4D;
    bias: tf.Tensor1D;
}

export declare const loadAgeGenderModel: (url: string) => Promise<void>;

export declare const loadFaceDetectionModel: (url: string) => Promise<void>;

export declare const loadFaceExpressionModel: (url: string) => Promise<void>;

export declare const loadFaceLandmarkModel: (url: string) => Promise<void>;

export declare const loadFaceLandmarkTinyModel: (url: string) => Promise<void>;

export declare const loadFaceRecognitionModel: (url: string) => Promise<void>;

export declare const loadSsdMobilenetv1Model: (url: string) => Promise<void>;

export declare const loadTinyFaceDetectorModel: (url: string) => Promise<void>;

export declare const loadTinyYolov2Model: (url: string) => Promise<void>;

export declare function loadWeightMap(uri: string | undefined, defaultModelName: string): Promise<tf.NamedTensorMap>;

export declare const locateFaces: (input: TNetInput, options: SsdMobilenetv1Options) => Promise<FaceDetection[]>;

declare type MainBlockParams = {
    separable_conv0: SeparableConvParams;
    separable_conv1: SeparableConvParams;
    separable_conv2: SeparableConvParams;
};

export declare function matchDimensions(input: IDimensions, reference: IDimensions, useMediaDimensions?: boolean): {
    width: number;
    height: number;
};

export declare function minBbox(pts: IPoint[]): BoundingBox;

export declare type MobilenetParams = {
    conv0: SeparableConvParams | ConvParams;
    conv1: SeparableConvParams;
    conv2: SeparableConvParams;
    conv3: SeparableConvParams;
    conv4: SeparableConvParams;
    conv5: SeparableConvParams;
    conv6?: SeparableConvParams;
    conv7?: SeparableConvParams;
    conv8: ConvParams;
};

declare namespace MobileNetV1 {
    type DepthwiseConvParams = {
        filters: tf.Tensor4D;
        batch_norm_scale: tf.Tensor1D;
        batch_norm_offset: tf.Tensor1D;
        batch_norm_mean: tf.Tensor1D;
        batch_norm_variance: tf.Tensor1D;
    };
    type ConvPairParams = {
        depthwise_conv: DepthwiseConvParams;
        pointwise_conv: PointwiseConvParams;
    };
    type Params = {
        conv_0: PointwiseConvParams;
        conv_1: ConvPairParams;
        conv_2: ConvPairParams;
        conv_3: ConvPairParams;
        conv_4: ConvPairParams;
        conv_5: ConvPairParams;
        conv_6: ConvPairParams;
        conv_7: ConvPairParams;
        conv_8: ConvPairParams;
        conv_9: ConvPairParams;
        conv_10: ConvPairParams;
        conv_11: ConvPairParams;
        conv_12: ConvPairParams;
        conv_13: ConvPairParams;
    };
}

declare function monkeyPatch(env: Partial<Environment>): void;

export declare class NetInput {
    private _imageTensors;
    private _canvases;
    private _batchSize;
    private _treatAsBatchInput;
    private _inputDimensions;
    private _inputSize;
    constructor(inputs: Array<TResolvedNetInput>, treatAsBatchInput?: boolean);
    get imageTensors(): Array<tf.Tensor3D | tf.Tensor4D>;
    get canvases(): HTMLCanvasElement[];
    get isBatchInput(): boolean;
    get batchSize(): number;
    get inputDimensions(): number[][];
    get inputSize(): number | undefined;
    get reshapedInputDimensions(): Dimensions[];
    getInput(batchIdx: number): tf.Tensor3D | tf.Tensor4D | HTMLCanvasElement;
    getInputDimensions(batchIdx: number): number[];
    getInputHeight(batchIdx: number): number;
    getInputWidth(batchIdx: number): number;
    getReshapedInputDimensions(batchIdx: number): Dimensions;
    /**
     * Create a batch tensor from all input canvases and tensors
     * with size [batchSize, inputSize, inputSize, 3].
     *
     * @param inputSize Height and width of the tensor.
     * @param isCenterImage (optional, default: false) If true, add an equal amount of padding on
     * both sides of the minor dimension oof the image.
     * @returns The batch tensor.
     */
    toBatchTensor(inputSize: number, isCenterInputs?: boolean): tf.Tensor4D;
}

export declare type NetOutput = {
    age: tf.Tensor1D;
    gender: tf.Tensor2D;
};

export declare type NetParams = {
    fc: {
        age: FCParams_2;
        gender: FCParams_2;
    };
};

declare type NetParams_2 = {
    fc: FCParams_2;
};

declare type NetParams_3 = {
    conv32_down: ConvLayerParams;
    conv32_1: ResidualLayerParams;
    conv32_2: ResidualLayerParams;
    conv32_3: ResidualLayerParams;
    conv64_down: ResidualLayerParams;
    conv64_1: ResidualLayerParams;
    conv64_2: ResidualLayerParams;
    conv64_3: ResidualLayerParams;
    conv128_down: ResidualLayerParams;
    conv128_1: ResidualLayerParams;
    conv128_2: ResidualLayerParams;
    conv256_down: ResidualLayerParams;
    conv256_1: ResidualLayerParams;
    conv256_2: ResidualLayerParams;
    conv256_down_out: ResidualLayerParams;
    fc: tf.Tensor2D;
};

declare type NetParams_4 = {
    mobilenetv1: MobileNetV1.Params;
    prediction_layer: PredictionLayerParams;
    output_layer: OutputLayerParams;
};

/**
 * BlazeFace network parameters.
 */
declare interface NetParams_5 {
    conv1: ConvBlockParams;
    conv2: ConvBlockParams;
    conv3: ConvBlockParams;
    conv4: ConvBlockParams;
    conv5: ConvBlockParams;
    conv6: ConvBlockParams;
    conv7: ConvBlockParams;
    conv8: ConvBlockParams;
    classifierHead: HeadParams;
    regressorHead: HeadParams;
}

/**
 * AdaFace network parameters.
 */
declare interface NetParams_6 {
    stem: StemParams;
    stage1: IRSEBlockParams;
    stage2: IRSEBlockParams;
    stage3: IRSEBlockParams;
    stage4: IRSEBlockParams;
    fc: FCParams;
}

/**
 * FaceMesh network parameters.
 */
declare interface NetParams_7 {
    encoder: EncoderParams;
    bottleneck: ResidualBlockParams[];
    decoder: DecoderParams;
    landmarkHead: LandmarkHeadParams;
}

export declare const nets: {
    ssdMobilenetv1: SsdMobilenetv1;
    tinyFaceDetector: TinyFaceDetector;
    tinyYolov2: TinyYolov2;
    faceLandmark68Net: FaceLandmark68Net;
    faceLandmark68TinyNet: FaceLandmark68TinyNet;
    faceRecognitionNet: FaceRecognitionNet;
    faceExpressionNet: FaceExpressionNet;
    ageGenderNet: AgeGenderNet;
};

export declare abstract class NeuralNetwork<TNetParams> {
    constructor(name: string);
    protected _params: TNetParams | undefined;
    protected _paramMappings: ParamMapping[];
    _name: string;
    get params(): TNetParams | undefined;
    get paramMappings(): ParamMapping[];
    get isLoaded(): boolean;
    getParamFromPath(paramPath: string): tf.Tensor;
    reassignParamFromPath(paramPath: string, tensor: tf.Tensor): void;
    getParamList(): {
        path: string;
        tensor: tf.Tensor<tf.Rank>;
    }[];
    getTrainableParams(): {
        path: string;
        tensor: tf.Tensor<tf.Rank>;
    }[];
    getFrozenParams(): {
        path: string;
        tensor: tf.Tensor<tf.Rank>;
    }[];
    variable(): void;
    freeze(): Promise<void>;
    dispose(throwOnRedispose?: boolean): void;
    /**
     * Serialize parameters to Float32Array (legacy format).
     * @deprecated Use serializeWithMetadata() for versioned output
     */
    serializeParams(): Promise<Float32Array>;
    /**
     * Serialize parameters with version and metadata.
     */
    serializeWithMetadata(metadata?: SerializedWeights['metadata']): Promise<SerializedWeights>;
    /**
     * Load parameters from versioned format.
     */
    loadFromSerialized(data: SerializedWeights): void;
    /**
     * Convert base64 string to Float32Array.
     */
    private _base64ToFloat32Array;
    load(weightsOrUrl: Float32Array | string | undefined): Promise<void>;
    loadFromUri(uri: string | undefined): Promise<void>;
    loadFromDisk(filePath: string | undefined): Promise<void>;
    loadFromWeightMap(weightMap: tf.NamedTensorMap): void;
    extractWeights(weights: Float32Array): void;
    private traversePropertyPath;
    protected abstract getDefaultModelName(): string;
    protected abstract extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TNetParams;
        paramMappings: ParamMapping[];
    };
    protected abstract extractParams(weights: Float32Array): {
        params: TNetParams;
        paramMappings: ParamMapping[];
    };
}

export declare function nonMaxSuppression(boxes: Box[], scores: number[], iouThreshold: number, isIOU?: boolean): number[];

/**
 * Normalize an image tensor by subtracting mean RGB values.
 * Optimized to use a single tensor operation with broadcasting.
 *
 * @param x Input tensor of shape [batch, height, width, 3]
 * @param meanRgb Array of [R, G, B] mean values to subtract
 * @returns Normalized tensor with same shape as input
 */
export declare function normalize(x: tf.Tensor4D, meanRgb: number[]): tf.Tensor4D;

export declare class ObjectDetection {
    private _score;
    private _classScore;
    private _className;
    private _box;
    private _imageDims;
    constructor(score: number, classScore: number, className: string, relativeBox: IRect, imageDims: IDimensions);
    get score(): number;
    get classScore(): number;
    get className(): string;
    get box(): Box;
    get imageDims(): Dimensions;
    get imageWidth(): number;
    get imageHeight(): number;
    get relativeBox(): Box;
    forSize(width: number, height: number): ObjectDetection;
}

declare type OutputLayerParams = {
    extra_dim: tf.Tensor3D;
};

/**
 * Pads the smaller dimension of an image tensor with zeros, such that width === height.
 * Uses tf.pad() for better performance instead of creating separate zero tensors.
 *
 * @param imgTensor The image tensor.
 * @param isCenterImage (optional, default: false) If true, add an equal amount of padding on
 * both sides of the minor dimension of the image.
 * @returns The padded tensor with width === height.
 */
export declare function padToSquare(imgTensor: tf.Tensor4D, isCenterImage?: boolean): tf.Tensor4D;

declare type ParamMapping = {
    originalPath?: string;
    paramPath: string;
};

export declare class Point implements IPoint {
    private _x;
    private _y;
    constructor(x: number, y: number);
    get x(): number;
    get y(): number;
    add(pt: IPoint): Point;
    sub(pt: IPoint): Point;
    mul(pt: IPoint): Point;
    div(pt: IPoint): Point;
    abs(): Point;
    magnitude(): number;
    floor(): Point;
}

declare type PointwiseConvParams = {
    filters: tf.Tensor4D;
    batch_norm_offset: tf.Tensor1D;
};

/**
 * Predicts age and gender from a face image.
 *
 * @param inputs The face image extracted from the bounding box of a face. Can
 * also be an array of input images, which will be batch processed.
 * @returns Predictions with age, gender and gender probability or array thereof in case of batch input.
 */
export declare const predictAgeAndGender: (input: TNetInput) => Promise<AgeAndGenderPrediction | AgeAndGenderPrediction[]>;

declare class PredictAgeAndGenderTaskBase<TReturn, TParentReturn> extends ComposableTask<TReturn> {
    protected parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>;
    protected input: TNetInput;
    protected extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | undefined;
    constructor(parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>, input: TNetInput, extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | undefined);
}

declare class PredictAllAgeAndGenderTask<TSource extends WithFaceDetection<{}>> extends PredictAgeAndGenderTaskBase<WithAge<WithGender<TSource>>[], TSource[]> {
    run(): Promise<WithAge<WithGender<TSource>>[]>;
    withFaceExpressions(): PredictAllFaceExpressionsTask<WithAge<WithGender<TSource>>>;
}

declare class PredictAllAgeAndGenderWithFaceAlignmentTask<TSource extends WithFaceLandmarks<WithFaceDetection<{}>>> extends PredictAllAgeAndGenderTask<TSource> {
    withFaceExpressions(): PredictAllFaceExpressionsWithFaceAlignmentTask<WithAge<WithGender<TSource>>>;
    withFaceDescriptors(): ComputeAllFaceDescriptorsTask<WithAge<WithGender<TSource>>>;
}

declare class PredictAllFaceExpressionsTask<TSource extends WithFaceDetection<{}>> extends PredictFaceExpressionsTaskBase<WithFaceExpressions<TSource>[], TSource[]> {
    run(): Promise<WithFaceExpressions<TSource>[]>;
    withAgeAndGender(): PredictAllAgeAndGenderTask<WithFaceExpressions<TSource>>;
}

declare class PredictAllFaceExpressionsWithFaceAlignmentTask<TSource extends WithFaceLandmarks<WithFaceDetection<{}>>> extends PredictAllFaceExpressionsTask<TSource> {
    withAgeAndGender(): PredictAllAgeAndGenderWithFaceAlignmentTask<WithFaceExpressions<TSource>>;
    withFaceDescriptors(): ComputeAllFaceDescriptorsTask<WithFaceExpressions<TSource>>;
}

export declare class PredictedBox extends LabeledBox {
    static assertIsValidPredictedBox(box: any, callee: string): void;
    private _score;
    private _classScore;
    constructor(box: IBoundingBox | IRect | any, label: number, score: number, classScore: number);
    get score(): number;
    get classScore(): number;
}

declare class PredictFaceExpressionsTaskBase<TReturn, TParentReturn> extends ComposableTask<TReturn> {
    protected parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>;
    protected input: TNetInput;
    protected extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | undefined;
    constructor(parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>, input: TNetInput, extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | undefined);
}

declare type PredictionLayerParams = {
    conv_0: PointwiseConvParams;
    conv_1: PointwiseConvParams;
    conv_2: PointwiseConvParams;
    conv_3: PointwiseConvParams;
    conv_4: PointwiseConvParams;
    conv_5: PointwiseConvParams;
    conv_6: PointwiseConvParams;
    conv_7: PointwiseConvParams;
    box_predictor_0: BoxPredictionParams;
    box_predictor_1: BoxPredictionParams;
    box_predictor_2: BoxPredictionParams;
    box_predictor_3: BoxPredictionParams;
    box_predictor_4: BoxPredictionParams;
    box_predictor_5: BoxPredictionParams;
};

declare class PredictSingleAgeAndGenderTask<TSource extends WithFaceDetection<{}>> extends PredictAgeAndGenderTaskBase<WithAge<WithGender<TSource>> | undefined, TSource | undefined> {
    run(): Promise<WithAge<WithGender<TSource>> | undefined>;
    withFaceExpressions(): PredictSingleFaceExpressionsTask<WithAge<WithGender<TSource>>>;
}

declare class PredictSingleAgeAndGenderWithFaceAlignmentTask<TSource extends WithFaceLandmarks<WithFaceDetection<{}>>> extends PredictSingleAgeAndGenderTask<TSource> {
    withFaceExpressions(): PredictSingleFaceExpressionsWithFaceAlignmentTask<WithAge<WithGender<TSource>>>;
    withFaceDescriptor(): ComputeSingleFaceDescriptorTask<WithAge<WithGender<TSource>>>;
}

declare class PredictSingleFaceExpressionsTask<TSource extends WithFaceDetection<{}>> extends PredictFaceExpressionsTaskBase<WithFaceExpressions<TSource> | undefined, TSource | undefined> {
    run(): Promise<WithFaceExpressions<TSource> | undefined>;
    withAgeAndGender(): PredictSingleAgeAndGenderTask<WithFaceExpressions<TSource>>;
}

declare class PredictSingleFaceExpressionsWithFaceAlignmentTask<TSource extends WithFaceLandmarks<WithFaceDetection<{}>>> extends PredictSingleFaceExpressionsTask<TSource> {
    withAgeAndGender(): PredictSingleAgeAndGenderWithFaceAlignmentTask<WithFaceExpressions<TSource>>;
    withFaceDescriptor(): ComputeSingleFaceDescriptorTask<WithFaceExpressions<TSource>>;
}

declare function range(num: number, start: number, step: number): number[];

/**
 * Recognizes the facial expressions from a face image.
 *
 * @param inputs The face image extracted from the bounding box of a face. Can
 * also be an array of input images, which will be batch processed.
 * @returns Facial expressions with corresponding probabilities or array thereof in case of batch input.
 */
export declare const recognizeFaceExpressions: (input: TNetInput) => Promise<FaceExpressions | FaceExpressions[]>;

export declare class Rect extends Box implements IRect {
    constructor(x: number, y: number, width: number, height: number, allowNegativeDimensions?: boolean);
}

declare type ReductionBlockParams = {
    separable_conv0: SeparableConvParams;
    separable_conv1: SeparableConvParams;
    expansion_conv: ConvParams;
};

/**
 * Reset backend state (for testing).
 */
export declare function resetBackendState(): void;

/**
 * Residual block parameters.
 */
export declare interface ResidualBlockParams {
    conv1: FaceMeshConvBlockParams;
    conv2: FaceMeshConvBlockParams;
}

declare type ResidualLayerParams = {
    conv1: ConvLayerParams;
    conv2: ConvLayerParams;
};

export declare function resizeResults<T>(results: T, dimensions: IDimensions): T;

export declare function resolveInput(arg: string | any): any;

declare function round(num: number, prec?: number): number;

declare type ScaleLayerParams = {
    weights: tf.Tensor1D;
    biases: tf.Tensor1D;
};

declare class SeparableConvParams {
    depthwise_filter: tf.Tensor4D;
    pointwise_filter: tf.Tensor4D;
    bias: tf.Tensor1D;
    constructor(depthwise_filter: tf.Tensor4D, pointwise_filter: tf.Tensor4D, bias: tf.Tensor1D);
}

/**
 * Squeeze-Excitation parameters.
 */
export declare interface SEParams {
    fc1: tf.Tensor2D;
    fc2: tf.Tensor2D;
}

/**
 * Serialized model weights with metadata.
 */
export declare interface SerializedWeights {
    /** Format version for forward compatibility */
    version: number;
    /** Model name */
    name: string;
    /** Total number of parameters */
    numParams: number;
    /** Parameter shapes for validation */
    shapes: Record<string, number[]>;
    /** Serialized weights as base64 or array */
    weights: string | number[];
    /** Optional training metadata */
    metadata?: {
        trainedAt?: string;
        epochs?: number;
        finalLoss?: number;
        framework?: string;
    };
    /** Checksum for integrity validation */
    checksum?: string;
}

declare function setEnv(env: Environment): void;

export declare function shuffleArray(inputArray: any[]): any[];

export declare function sigmoid(x: number): number;

/**
 * Analysis result for a specific image source.
 */
export declare interface SourceAnalysisResult {
    /** Type of source analyzed */
    sourceType: ImageSourceType;
    /** Was a face detected? */
    faceDetected: boolean;
    /** Detection result */
    detection: FaceDetection | null;
    /** Landmarks (68-point) */
    landmarks: FaceLandmarks68 | null;
    /** BlazeFace keypoints (6-point: eyes, ears, nose, mouth) */
    keypoints: BlazeFaceKeypoints | null;
    /** Face descriptor (512-dim for AdaFace, 128-dim for FaceNet fallback) */
    descriptor: Float32Array | null;
    /** Quality score (0-1) */
    qualityScore: number;
    /** Detailed quality breakdown */
    quality: {
        sharpness: number;
        brightness: number;
        contrast: number;
        faceSize: number;
        frontalScore: number;
        overallConfidence: number;
    };
    /** Estimated image issues */
    estimatedIssues: string[];
    /** Is acceptable for this source type? */
    isAcceptable: boolean;
    /** Model info for debugging */
    modelInfo: {
        detector: 'blazeface' | 'ssd_mobilenetv1' | 'tiny_face_detector';
        recognizer: 'adaface' | 'facenet';
        descriptorDim: number;
    };
}

export declare class SsdMobilenetv1 extends NeuralNetwork<NetParams_4> {
    constructor();
    forwardInput(input: NetInput): {
        boxes: tf.Tensor2D[];
        scores: tf.Tensor1D[];
    };
    forward(input: TNetInput): Promise<{
        boxes: tf.Tensor2D[];
        scores: tf.Tensor1D[];
    }>;
    locateFaces(input: TNetInput, options?: ISsdMobilenetv1Options): Promise<FaceDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: NetParams_4;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: NetParams_4;
        paramMappings: ParamMapping[];
    };
}

/**
 * Attempts to detect all faces in an image using SSD Mobilenetv1 Network.
 *
 * @param input The input image.
 * @param options (optional, default: see SsdMobilenetv1Options constructor for default parameters).
 * @returns Bounding box of each face with score.
 */
export declare const ssdMobilenetv1: (input: TNetInput, options: SsdMobilenetv1Options) => Promise<FaceDetection[]>;

export declare class SsdMobilenetv1Options {
    protected _name: string;
    private _minConfidence;
    private _maxResults;
    constructor({ minConfidence, maxResults }?: ISsdMobilenetv1Options);
    get minConfidence(): number;
    get maxResults(): number;
}

/**
 * Stem parameters.
 */
export declare interface StemParams {
    conv1: ConvBnParams;
    conv2: ConvBnParams;
    conv3: ConvBnParams;
}

declare type TDrawDetectionsInput = IRect | IBoundingBox | FaceDetection | WithFaceDetection<{}>;

export { tf }

export declare class TinyFaceDetector extends TinyYolov2Base {
    constructor();
    get anchors(): Point[];
    locateFaces(input: TNetInput, forwardParams: ITinyYolov2Options): Promise<FaceDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TinyYolov2NetParams;
        paramMappings: ParamMapping[];
    };
}

/**
 * Attempts to detect all faces in an image using the Tiny Face Detector.
 *
 * @param input The input image.
 * @param options (optional, default: see TinyFaceDetectorOptions constructor for default parameters).
 * @returns Bounding box of each face with score.
 */
export declare const tinyFaceDetector: (input: TNetInput, options: TinyFaceDetectorOptions) => Promise<FaceDetection[]>;

export declare class TinyFaceDetectorOptions extends TinyYolov2Options {
    protected _name: string;
}

declare class TinyFaceFeatureExtractor extends NeuralNetwork<TinyFaceFeatureExtractorParams> implements IFaceFeatureExtractor<TinyFaceFeatureExtractorParams> {
    constructor();
    forwardInput(input: NetInput): tf.Tensor4D;
    forward(input: TNetInput): Promise<tf.Tensor4D>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TinyFaceFeatureExtractorParams;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: TinyFaceFeatureExtractorParams;
        paramMappings: ParamMapping[];
    };
}

declare type TinyFaceFeatureExtractorParams = {
    dense0: DenseBlock3Params;
    dense1: DenseBlock3Params;
    dense2: DenseBlock3Params;
};

declare class TinyXception extends NeuralNetwork<TinyXceptionParams> {
    private _numMainBlocks;
    constructor(numMainBlocks: number);
    forwardInput(input: NetInput): tf.Tensor4D;
    forward(input: TNetInput): Promise<tf.Tensor4D>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TinyXceptionParams;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: TinyXceptionParams;
        paramMappings: ParamMapping[];
    };
}

declare type TinyXceptionParams = {
    entry_flow: {
        conv_in: ConvParams;
        reduction_block_0: ReductionBlockParams;
        reduction_block_1: ReductionBlockParams;
    };
    middle_flow: Record<`main_block_${number}`, MainBlockParams>;
    exit_flow: {
        reduction_block: ReductionBlockParams;
        separable_conv: SeparableConvParams;
    };
};

export declare class TinyYolov2 extends TinyYolov2Base {
    constructor(withSeparableConvs?: boolean);
    get withSeparableConvs(): boolean;
    get anchors(): Point[];
    locateFaces(input: TNetInput, forwardParams: ITinyYolov2Options): Promise<FaceDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TinyYolov2NetParams;
        paramMappings: ParamMapping[];
    };
}

/**
 * Attempts to detect all faces in an image using the Tiny Yolov2 Network.
 *
 * @param input The input image.
 * @param options (optional, default: see TinyYolov2Options constructor for default parameters).
 * @returns Bounding box of each face with score.
 */
export declare const tinyYolov2: (input: TNetInput, options: ITinyYolov2Options) => Promise<FaceDetection[]>;

declare class TinyYolov2Base extends NeuralNetwork<TinyYolov2NetParams> {
    static DEFAULT_FILTER_SIZES: number[];
    private _config;
    constructor(config: TinyYolov2Config);
    get config(): TinyYolov2Config;
    get withClassScores(): boolean;
    get boxEncodingSize(): number;
    runTinyYolov2(x: tf.Tensor4D, params: DefaultTinyYolov2NetParams): tf.Tensor4D;
    runMobilenet(x: tf.Tensor4D, params: MobilenetParams): tf.Tensor4D;
    forwardInput(input: NetInput, inputSize: number): tf.Tensor4D;
    forward(input: TNetInput, inputSize: number): Promise<tf.Tensor4D>;
    detect(input: TNetInput, forwardParams?: ITinyYolov2Options): Promise<ObjectDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TinyYolov2NetParams;
        paramMappings: ParamMapping[];
    };
    protected extractParams(weights: Float32Array): {
        params: TinyYolov2NetParams;
        paramMappings: ParamMapping[];
    };
    protected extractBoxes(outputTensor: tf.Tensor4D, inputBlobDimensions: Dimensions, scoreThreshold?: number): Promise<TinyYolov2ExtractBoxesResult[]>;
    private extractPredictedClass;
}

export declare type TinyYolov2Config = {
    withSeparableConvs: boolean;
    iouThreshold: number;
    anchors: Point[];
    classes: string[];
    meanRgb?: [number, number, number];
    withClassScores?: boolean;
    filterSizes?: number[];
    isFirstLayerConv2d?: boolean;
};

export declare type TinyYolov2ExtractBoxesResult = {
    box: BoundingBox;
    score: number;
    classScore: number;
    label: number;
    row: number;
    col: number;
    anchor: number;
};

export declare type TinyYolov2NetParams = DefaultTinyYolov2NetParams | MobilenetParams;

export declare class TinyYolov2Options {
    protected _name: string;
    private _inputSize;
    private _scoreThreshold;
    constructor({ inputSize, scoreThreshold }?: ITinyYolov2Options);
    get inputSize(): number;
    get scoreThreshold(): number;
}

export declare type TMediaElement = HTMLImageElement | HTMLVideoElement | HTMLCanvasElement;

export declare type TNetInput = string | TResolvedNetInput | Array<string | TResolvedNetInput> | NetInput;

/**
 * Validates the input to make sure, they are valid net inputs and awaits all media elements
 * to be finished loading.
 *
 * @param input The input, which can be a media element or an array of different media elements.
 * @returns A NetInput instance, which can be passed into one of the neural networks.
 */
export declare function toNetInput(inputs: TNetInput): Promise<NetInput>;

export declare type TResolvedNetInput = TMediaElement | Tensor3D | Tensor4D;

declare namespace utils {
    export {
        isTensor,
        isTensor1D,
        isTensor2D,
        isTensor3D,
        isTensor4D,
        isFloat,
        isEven,
        round,
        isDimensions,
        computeReshapedDimensions,
        getCenterPoint,
        range,
        isValidNumber,
        isValidProbablitiy
    }
}
export { utils }

export declare function validateConfig(config: any): void;

export declare const version: string;

/**
 * WebGPU backend initialization and utilities.
 * Provides helpers for setting up WebGPU with face-api.js.
 */
export declare interface WebGPUInfo {
    supported: boolean;
    adapter: GPUAdapter | null;
    device: GPUDevice | null;
    adapterInfo: GPUAdapterInfo | null;
}

export declare type WithAge<TSource> = TSource & {
    age: number;
};

export declare type WithFaceDescriptor<TSource> = TSource & {
    descriptor: Float32Array;
};

export declare type WithFaceDetection<TSource> = TSource & {
    detection: FaceDetection;
};

export declare type WithFaceExpressions<TSource> = TSource & {
    expressions: FaceExpressions;
};

export declare type WithFaceLandmarks<TSource extends WithFaceDetection<{}>, TFaceLandmarks extends FaceLandmarks = FaceLandmarks68> = TSource & {
    landmarks: TFaceLandmarks;
    unshiftedLandmarks: TFaceLandmarks;
    alignedRect: FaceDetection;
    angle: {
        roll: number | undefined;
        pitch: number | undefined;
        yaw: number | undefined;
    };
};

export declare type WithGender<TSource> = TSource & {
    gender: Gender;
    genderProbability: number;
};

export { }

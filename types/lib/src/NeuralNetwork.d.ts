import * as tf from '@tensorflow/tfjs';
import type { ParamMapping } from './common/index';
/**
 * Serialized model weights with metadata.
 */
export interface SerializedWeights {
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

import * as tf from '@tensorflow/tfjs';

import type { ParamMapping } from './common/index';
import { getModelUris } from './common/getModelUris';
import { loadWeightMap } from './dom/index';
import { env } from './env/index';

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

export abstract class NeuralNetwork<TNetParams> {
  constructor(name: string) {
    this._name = name;
  }

  protected _params: TNetParams | undefined = undefined;

  protected _paramMappings: ParamMapping[] = [];

  public _name: string;

  public get params(): TNetParams | undefined { return this._params; }

  public get paramMappings(): ParamMapping[] { return this._paramMappings; }

  public get isLoaded(): boolean { return !!this.params; }

  public getParamFromPath(paramPath: string): tf.Tensor {
    const { obj, objProp } = this.traversePropertyPath(paramPath);
    return obj[objProp];
  }

  public reassignParamFromPath(paramPath: string, tensor: tf.Tensor) {
    const { obj, objProp } = this.traversePropertyPath(paramPath);
    obj[objProp].dispose();
    obj[objProp] = tensor;
  }

  public getParamList() {
    return this._paramMappings.map(({ paramPath }) => ({
      path: paramPath,
      tensor: this.getParamFromPath(paramPath),
    }));
  }

  public getTrainableParams() {
    return this.getParamList().filter((param) => param.tensor instanceof tf.Variable);
  }

  public getFrozenParams() {
    return this.getParamList().filter((param) => !(param.tensor instanceof tf.Variable));
  }

  public variable() {
    this.getFrozenParams().forEach(({ path, tensor }) => {
      this.reassignParamFromPath(path, tensor.variable());
    });
  }

  public async freeze(): Promise<void> {
    const trainableParams = this.getTrainableParams();
    for (const { path, tensor: variable } of trainableParams) {
      const data = await variable.data();
      const tensor = tf.tensor(data, variable.shape);
      variable.dispose();
      this.reassignParamFromPath(path, tensor);
    }
  }

  public dispose(throwOnRedispose = true) {
    this.getParamList().forEach((param) => {
      if (throwOnRedispose && param.tensor.isDisposed) {
        throw new Error(`param tensor has already been disposed for path ${param.path}`);
      }
      param.tensor.dispose();
    });
    this._params = undefined;
  }

  /**
   * Serialize parameters to Float32Array (legacy format).
   * @deprecated Use serializeWithMetadata() for versioned output
   */
  public async serializeParams(): Promise<Float32Array> {
    const paramList = this.getParamList();
    const arrays: number[] = [];

    for (const { tensor } of paramList) {
      const data = await tensor.data();
      arrays.push(...Array.from(data));
    }

    return new Float32Array(arrays);
  }

  /**
   * Serialize parameters with version and metadata.
   */
  public async serializeWithMetadata(metadata?: SerializedWeights['metadata']): Promise<SerializedWeights> {
    const paramList = this.getParamList();
    const shapes: Record<string, number[]> = {};
    const arrays: number[] = [];

    for (const { path, tensor } of paramList) {
      shapes[path] = Array.from(tensor.shape);
      const data = await tensor.data();
      arrays.push(...Array.from(data));
    }

    // Simple checksum: sum of first 100 values
    const checksum = arrays.slice(0, 100).reduce((a, b) => a + b, 0).toFixed(6);

    return {
      version: 1,
      name: this._name,
      numParams: arrays.length,
      shapes,
      weights: arrays,
      metadata: {
        ...metadata,
        framework: 'face-api.js',
        trainedAt: new Date().toISOString(),
      },
      checksum,
    };
  }

  /**
   * Load parameters from versioned format.
   */
  public loadFromSerialized(data: SerializedWeights): void {
    if (data.version !== 1) {
      throw new Error(`Unsupported weight format version: ${data.version}`);
    }

    if (data.name !== this._name) {
      console.warn(`Model name mismatch: expected ${this._name}, got ${data.name}`);
    }

    const weights = Array.isArray(data.weights)
      ? new Float32Array(data.weights)
      : this._base64ToFloat32Array(data.weights);

    // Validate checksum if present
    if (data.checksum) {
      const checksum = Array.from(weights).slice(0, 100).reduce((a, b) => a + b, 0).toFixed(6);
      if (checksum !== data.checksum) {
        throw new Error('Weight checksum mismatch - data may be corrupted');
      }
    }

    this.extractWeights(weights);
  }

  /**
   * Convert base64 string to Float32Array.
   */
  private _base64ToFloat32Array(base64: string): Float32Array {
    if (typeof atob !== 'undefined') {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      return new Float32Array(bytes.buffer);
    }
    // Node.js
    const buffer = Buffer.from(base64, 'base64');
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
  }

  public async load(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    if (weightsOrUrl instanceof Float32Array) {
      this.extractWeights(weightsOrUrl);
      return;
    }
    await this.loadFromUri(weightsOrUrl);
  }

  public async loadFromUri(uri: string | undefined) {
    if (uri && typeof uri !== 'string') {
      throw new Error(`${this._name}.loadFromUri - expected model uri`);
    }
    const weightMap = await loadWeightMap(uri, this.getDefaultModelName());
    this.loadFromWeightMap(weightMap);
  }

  public async loadFromDisk(filePath: string | undefined) {
    if (filePath && typeof filePath !== 'string') {
      throw new Error(`${this._name}.loadFromDisk - expected model file path`);
    }
    const { readFile } = env.getEnv();
    const { manifestUri, modelBaseUri } = getModelUris(filePath, this.getDefaultModelName());

    // Fetch weights from disk - returns ArrayBuffer for each file path
    const fetchWeightsFromDisk = async (filePaths: string[]): Promise<ArrayBuffer[]> => Promise.all(
      filePaths.map(async (fp) => {
        const buf = await readFile(fp);
        if (typeof buf === 'string') {
          return Buffer.from(buf).buffer;
        }
        return buf.buffer;
      }),
    );

    // Type for internal TensorFlow.js weightsLoaderFactory API
    type WeightsLoaderFactory = (
      fetchWeightsFunction: (filePaths: string[]) => Promise<ArrayBuffer[]>,
    ) => (
      manifest: tf.io.WeightsManifestConfig,
      baseUri: string,
    ) => Promise<tf.NamedTensorMap>;

    // Access internal weightsLoaderFactory (used for custom weight loading)
    const tfIO = tf.io as typeof tf.io & { weightsLoaderFactory: WeightsLoaderFactory };
    const loadWeights = tfIO.weightsLoaderFactory(fetchWeightsFromDisk);

    const manifest = JSON.parse((await readFile(manifestUri)).toString()) as tf.io.WeightsManifestConfig;
    const weightMap = await loadWeights(manifest, modelBaseUri);
    this.loadFromWeightMap(weightMap);
  }

  public loadFromWeightMap(weightMap: tf.NamedTensorMap) {
    const { paramMappings, params } = this.extractParamsFromWeightMap(weightMap);
    this._paramMappings = paramMappings;
    this._params = params;
  }

  public extractWeights(weights: Float32Array) {
    const { paramMappings, params } = this.extractParams(weights);
    this._paramMappings = paramMappings;
    this._params = params;
  }

  private traversePropertyPath(paramPath: string) {
    if (!this.params) {
      throw new Error('traversePropertyPath - model has no loaded params');
    }

    const result = paramPath.split('/').reduce((res: { nextObj: any, obj?: any, objProp?: string }, objProp) => {
      // eslint-disable-next-line no-prototype-builtins
      if (!res.nextObj.hasOwnProperty(objProp)) {
        throw new Error(`traversePropertyPath - object does not have property ${objProp}, for path ${paramPath}`);
      }
      return { obj: res.nextObj, objProp, nextObj: res.nextObj[objProp] };
    }, { nextObj: this.params });

    const { obj, objProp } = result;
    if (!obj || !objProp || !(obj[objProp] instanceof tf.Tensor)) {
      throw new Error(`traversePropertyPath - parameter is not a tensor, for path ${paramPath}`);
    }

    return { obj, objProp };
  }

  protected abstract getDefaultModelName(): string

  // eslint-disable-next-line no-unused-vars
  protected abstract extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): { params: TNetParams, paramMappings: ParamMapping[] }

  // eslint-disable-next-line no-unused-vars
  protected abstract extractParams(weights: Float32Array): { params: TNetParams, paramMappings: ParamMapping[] }
}

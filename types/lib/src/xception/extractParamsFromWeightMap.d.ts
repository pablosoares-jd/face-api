import type * as tf from '@tensorflow/tfjs';
import type { ParamMapping } from '../common/index';
import type { TinyXceptionParams } from './types';
export declare function extractParamsFromWeightMap(weightMap: tf.NamedTensorMap, numMainBlocks: number): {
    params: TinyXceptionParams;
    paramMappings: ParamMapping[];
};

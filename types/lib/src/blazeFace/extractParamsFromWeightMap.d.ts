import type * as tf from '@tensorflow/tfjs';
import type { ParamMapping } from '../common/types';
import type { NetParams } from './types';
/**
 * Extract BlazeFace parameters from a TensorFlow.js weight map.
 */
export declare function extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
    params: NetParams;
    paramMappings: ParamMapping[];
};

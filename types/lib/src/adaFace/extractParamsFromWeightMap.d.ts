import * as tf from '@tensorflow/tfjs';
import { ParamMapping } from '../common/types';
import { NetParams } from './types';
/**
 * Extract AdaFace parameters from a TensorFlow.js weight map.
 */
export declare function extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
    params: NetParams;
    paramMappings: ParamMapping[];
};

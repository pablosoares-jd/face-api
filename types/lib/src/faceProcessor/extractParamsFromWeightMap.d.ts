import * as tf from '@tensorflow/tfjs';
import { ParamMapping } from '../common/index';
import { NetParams } from './types';
export declare function extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
    params: NetParams;
    paramMappings: ParamMapping[];
};

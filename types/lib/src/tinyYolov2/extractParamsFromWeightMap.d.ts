import type * as tf from '@tensorflow/tfjs';
import type { ParamMapping } from '../common/types';
import type { TinyYolov2Config } from './config';
import type { TinyYolov2NetParams } from './types';
export declare function extractParamsFromWeightMap(weightMap: tf.NamedTensorMap, config: TinyYolov2Config): {
    params: TinyYolov2NetParams;
    paramMappings: ParamMapping[];
};

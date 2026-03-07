import type * as tf from '@tensorflow/tfjs';
import type { ParamMapping } from '../common/index';
import type { FaceFeatureExtractorParams } from './types';
export declare function extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
    params: FaceFeatureExtractorParams;
    paramMappings: ParamMapping[];
};

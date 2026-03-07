import type * as tf from '@tensorflow/tfjs';
import type { ParamMapping } from '../common/index';
import type { TinyFaceFeatureExtractorParams } from './types';
export declare function extractParamsFromWeightMapTiny(weightMap: tf.NamedTensorMap): {
    params: TinyFaceFeatureExtractorParams;
    paramMappings: ParamMapping[];
};

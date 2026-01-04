import * as tf from '@tensorflow/tfjs';
import { ParamMapping } from '../common/index';
import { TinyFaceFeatureExtractorParams } from './types';
export declare function extractParamsFromWeightMapTiny(weightMap: tf.NamedTensorMap): {
    params: TinyFaceFeatureExtractorParams;
    paramMappings: ParamMapping[];
};

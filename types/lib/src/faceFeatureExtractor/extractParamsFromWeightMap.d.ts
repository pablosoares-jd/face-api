import * as tf from '@tensorflow/tfjs';
import { ParamMapping } from '../common/index';
import { FaceFeatureExtractorParams } from './types';
export declare function extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
    params: FaceFeatureExtractorParams;
    paramMappings: ParamMapping[];
};

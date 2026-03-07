import type { ParamMapping } from '../common/index';
import type { TinyFaceFeatureExtractorParams } from './types';
export declare function extractParamsTiny(weights: Float32Array): {
    params: TinyFaceFeatureExtractorParams;
    paramMappings: ParamMapping[];
};

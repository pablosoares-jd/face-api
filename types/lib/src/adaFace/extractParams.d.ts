import type { ParamMapping } from '../common/types';
import type { NetParams } from './types';
/**
 * Extract AdaFace parameters from weight array.
 * Note: This is a placeholder - actual weights need to be converted from PyTorch format.
 */
export declare function extractParams(weights: Float32Array): {
    params: NetParams;
    paramMappings: ParamMapping[];
};

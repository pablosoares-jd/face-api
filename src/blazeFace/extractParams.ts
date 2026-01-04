import { ParamMapping } from '../common/types';
import { NetParams, ConvBlockParams, HeadParams } from './types';

/**
 * Extract BlazeFace parameters from weight array.
 * Note: This is a placeholder - actual weights need to be converted from MediaPipe format.
 */
export function extractParams(weights: Float32Array): { params: NetParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  // Placeholder implementation - actual implementation would parse the weights
  throw new Error(
    'BlazeFace extractParams: Binary weight loading not supported. ' +
    'Use loadFromUri with a weight manifest instead.',
  );
}

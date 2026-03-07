import type { ParamMapping } from '../common/types';
import type { NetParams } from './types';

/**
 * Extract BlazeFace parameters from weight array.
 * Note: This is a placeholder - actual weights need to be converted from MediaPipe format.
 */
export function extractParams(weights: Float32Array): { params: NetParams; paramMappings: ParamMapping[] } {
  throw new Error(
    `BlazeFace extractParams: Binary weight loading not supported (received ${weights.length} weights). `
    + 'Use loadFromUri with a weight manifest instead.',
  );
}

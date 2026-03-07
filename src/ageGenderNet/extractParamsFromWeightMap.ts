import type * as tf from '@tensorflow/tfjs';

import type { FCParams, ParamMapping } from '../common/index';
import { disposeUnusedWeightTensors, extractWeightEntryFactory } from '../common/index';
import type { NetParams } from './types';

export function extractParamsFromWeightMap(
  weightMap: tf.NamedTensorMap,
): { params: NetParams, paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const extractWeightEntry = extractWeightEntryFactory(weightMap, paramMappings);

  function extractFcParams(prefix: string): FCParams {
    const weights = extractWeightEntry(`${prefix}/weights`, 2);
    const bias = extractWeightEntry(`${prefix}/bias`, 1);
    return { weights, bias };
  }

  const params = {
    fc: {
      age: extractFcParams('fc/age'),
      gender: extractFcParams('fc/gender'),
    },
  };

  disposeUnusedWeightTensors(weightMap, paramMappings);

  return { params, paramMappings };
}

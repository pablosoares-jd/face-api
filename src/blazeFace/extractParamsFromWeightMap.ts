import type * as tf from '@tensorflow/tfjs';

import type { ParamMapping } from '../common/types';
import type { NetParams, ConvBlockParams, HeadParams } from './types';

function extractConvBlockParams(
  weightMap: tf.NamedTensorMap,
  prefix: string,
  paramMappings: ParamMapping[],
): ConvBlockParams {
  const depthwise = weightMap[`${prefix}/depthwise/weights`] as tf.Tensor4D;
  const depthwiseBias = weightMap[`${prefix}/depthwise/bias`] as tf.Tensor1D;
  const pointwise = weightMap[`${prefix}/pointwise/weights`] as tf.Tensor4D;
  const pointwiseBias = weightMap[`${prefix}/pointwise/bias`] as tf.Tensor1D;

  paramMappings.push(
    { paramPath: `${prefix}/depthwise/weights` },
    { paramPath: `${prefix}/depthwise/bias` },
    { paramPath: `${prefix}/pointwise/weights` },
    { paramPath: `${prefix}/pointwise/bias` },
  );

  return { depthwise, depthwiseBias, pointwise, pointwiseBias };
}

function extractHeadParams(
  weightMap: tf.NamedTensorMap,
  prefix: string,
  paramMappings: ParamMapping[],
): HeadParams {
  const weights = weightMap[`${prefix}/weights`] as tf.Tensor4D;
  const bias = weightMap[`${prefix}/bias`] as tf.Tensor1D;

  paramMappings.push(
    { paramPath: `${prefix}/weights` },
    { paramPath: `${prefix}/bias` },
  );

  return { weights, bias };
}

/**
 * Extract BlazeFace parameters from a TensorFlow.js weight map.
 */
export function extractParamsFromWeightMap(
  weightMap: tf.NamedTensorMap,
): { params: NetParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const params: NetParams = {
    conv1: extractConvBlockParams(weightMap, 'conv1', paramMappings),
    conv2: extractConvBlockParams(weightMap, 'conv2', paramMappings),
    conv3: extractConvBlockParams(weightMap, 'conv3', paramMappings),
    conv4: extractConvBlockParams(weightMap, 'conv4', paramMappings),
    conv5: extractConvBlockParams(weightMap, 'conv5', paramMappings),
    conv6: extractConvBlockParams(weightMap, 'conv6', paramMappings),
    conv7: extractConvBlockParams(weightMap, 'conv7', paramMappings),
    conv8: extractConvBlockParams(weightMap, 'conv8', paramMappings),
    classifierHead: extractHeadParams(weightMap, 'classifier', paramMappings),
    regressorHead: extractHeadParams(weightMap, 'regressor', paramMappings),
  };

  return { params, paramMappings };
}

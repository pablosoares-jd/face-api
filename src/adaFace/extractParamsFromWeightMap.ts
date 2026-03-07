import type * as tf from '@tensorflow/tfjs';

import type { ParamMapping } from '../common/types';
import type { NetParams, ConvBnParams, IRSEBlockParams, SEParams, StemParams, FCParams } from './types';

function extractConvBnParams(
  weightMap: tf.NamedTensorMap,
  prefix: string,
  paramMappings: ParamMapping[],
  stride?: number,
): ConvBnParams {
  const weights = weightMap[`${prefix}/weights`] as tf.Tensor4D;
  const bn_mean = weightMap[`${prefix}/bn/mean`] as tf.Tensor1D;
  const bn_variance = weightMap[`${prefix}/bn/variance`] as tf.Tensor1D;
  const bn_offset = weightMap[`${prefix}/bn/offset`] as tf.Tensor1D;
  const bn_scale = weightMap[`${prefix}/bn/scale`] as tf.Tensor1D;

  paramMappings.push(
    { paramPath: `${prefix}/weights` },
    { paramPath: `${prefix}/bn/mean` },
    { paramPath: `${prefix}/bn/variance` },
    { paramPath: `${prefix}/bn/offset` },
    { paramPath: `${prefix}/bn/scale` },
  );

  return { weights, bn_mean, bn_variance, bn_offset, bn_scale, stride };
}

function extractSEParams(
  weightMap: tf.NamedTensorMap,
  prefix: string,
  paramMappings: ParamMapping[],
): SEParams {
  const fc1 = weightMap[`${prefix}/fc1`] as tf.Tensor2D;
  const fc2 = weightMap[`${prefix}/fc2`] as tf.Tensor2D;

  paramMappings.push(
    { paramPath: `${prefix}/fc1` },
    { paramPath: `${prefix}/fc2` },
  );

  return { fc1, fc2 };
}

function extractIRSEBlockParams(
  weightMap: tf.NamedTensorMap,
  prefix: string,
  paramMappings: ParamMapping[],
): IRSEBlockParams {
  const depthwise = weightMap[`${prefix}/depthwise`] as tf.Tensor4D;
  const bn1_mean = weightMap[`${prefix}/bn1/mean`] as tf.Tensor1D;
  const bn1_variance = weightMap[`${prefix}/bn1/variance`] as tf.Tensor1D;
  const bn1_offset = weightMap[`${prefix}/bn1/offset`] as tf.Tensor1D;
  const bn1_scale = weightMap[`${prefix}/bn1/scale`] as tf.Tensor1D;
  const se = extractSEParams(weightMap, `${prefix}/se`, paramMappings);
  const pointwise = weightMap[`${prefix}/pointwise`] as tf.Tensor4D;
  const bn2_mean = weightMap[`${prefix}/bn2/mean`] as tf.Tensor1D;
  const bn2_variance = weightMap[`${prefix}/bn2/variance`] as tf.Tensor1D;
  const bn2_offset = weightMap[`${prefix}/bn2/offset`] as tf.Tensor1D;
  const bn2_scale = weightMap[`${prefix}/bn2/scale`] as tf.Tensor1D;

  paramMappings.push(
    { paramPath: `${prefix}/depthwise` },
    { paramPath: `${prefix}/bn1/mean` },
    { paramPath: `${prefix}/bn1/variance` },
    { paramPath: `${prefix}/bn1/offset` },
    { paramPath: `${prefix}/bn1/scale` },
    { paramPath: `${prefix}/pointwise` },
    { paramPath: `${prefix}/bn2/mean` },
    { paramPath: `${prefix}/bn2/variance` },
    { paramPath: `${prefix}/bn2/offset` },
    { paramPath: `${prefix}/bn2/scale` },
  );

  return {
    depthwise,
    bn1_mean,
    bn1_variance,
    bn1_offset,
    bn1_scale,
    se,
    pointwise,
    bn2_mean,
    bn2_variance,
    bn2_offset,
    bn2_scale,
  };
}

/**
 * Extract AdaFace parameters from a TensorFlow.js weight map.
 */
export function extractParamsFromWeightMap(
  weightMap: tf.NamedTensorMap,
): { params: NetParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const stem: StemParams = {
    conv1: extractConvBnParams(weightMap, 'stem/conv1', paramMappings, 2),
    conv2: extractConvBnParams(weightMap, 'stem/conv2', paramMappings, 1),
    conv3: extractConvBnParams(weightMap, 'stem/conv3', paramMappings, 2),
  };

  const fc: FCParams = {
    weights: weightMap['fc/weights'] as tf.Tensor2D,
    bias: weightMap['fc/bias'] as tf.Tensor1D,
  };
  paramMappings.push({ paramPath: 'fc/weights' }, { paramPath: 'fc/bias' });

  const params: NetParams = {
    stem,
    stage1: extractIRSEBlockParams(weightMap, 'stage1', paramMappings),
    stage2: extractIRSEBlockParams(weightMap, 'stage2', paramMappings),
    stage3: extractIRSEBlockParams(weightMap, 'stage3', paramMappings),
    stage4: extractIRSEBlockParams(weightMap, 'stage4', paramMappings),
    fc,
  };

  return { params, paramMappings };
}

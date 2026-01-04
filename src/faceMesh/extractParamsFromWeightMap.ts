import * as tf from '../../dist/tfjs.esm';

import { ParamMapping } from '../common/types';
import {
  NetParams,
  ConvBlockParams,
  ResidualBlockParams,
  EncoderParams,
  DecoderParams,
  LandmarkHeadParams,
} from './types';

function extractConvBlockParams(
  weightMap: tf.NamedTensorMap,
  prefix: string,
  paramMappings: ParamMapping[],
): ConvBlockParams {
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

  return { weights, bn_mean, bn_variance, bn_offset, bn_scale };
}

function extractResidualBlockParams(
  weightMap: tf.NamedTensorMap,
  prefix: string,
  paramMappings: ParamMapping[],
): ResidualBlockParams {
  return {
    conv1: extractConvBlockParams(weightMap, `${prefix}/conv1`, paramMappings),
    conv2: extractConvBlockParams(weightMap, `${prefix}/conv2`, paramMappings),
  };
}

/**
 * Extract FaceMesh parameters from a TensorFlow.js weight map.
 */
export function extractParamsFromWeightMap(
  weightMap: tf.NamedTensorMap,
): { params: NetParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const encoder: EncoderParams = {
    conv1: extractConvBlockParams(weightMap, 'encoder/conv1', paramMappings),
    conv2: extractConvBlockParams(weightMap, 'encoder/conv2', paramMappings),
    conv3: extractConvBlockParams(weightMap, 'encoder/conv3', paramMappings),
    conv4: extractConvBlockParams(weightMap, 'encoder/conv4', paramMappings),
    conv5: extractConvBlockParams(weightMap, 'encoder/conv5', paramMappings),
  };

  // Extract bottleneck blocks (assume 4 blocks)
  const bottleneck: ResidualBlockParams[] = [];
  for (let i = 0; i < 4; i++) {
    bottleneck.push(extractResidualBlockParams(weightMap, `bottleneck/block${i}`, paramMappings));
  }

  const decoder: DecoderParams = {
    conv1: extractConvBlockParams(weightMap, 'decoder/conv1', paramMappings),
  };

  const landmarkHead: LandmarkHeadParams = {
    weights: weightMap['landmark_head/weights'] as tf.Tensor4D,
    bias: weightMap['landmark_head/bias'] as tf.Tensor1D,
  };
  paramMappings.push(
    { paramPath: 'landmark_head/weights' },
    { paramPath: 'landmark_head/bias' },
  );

  const params: NetParams = {
    encoder,
    bottleneck,
    decoder,
    landmarkHead,
  };

  return { params, paramMappings };
}

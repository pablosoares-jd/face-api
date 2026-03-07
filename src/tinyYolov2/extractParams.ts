import * as tf from '@tensorflow/tfjs';

import { extractConvParamsFactory } from '../common/index';
import { extractSeparableConvParamsFactory } from '../common/extractSeparableConvParamsFactory';
import { extractWeightsFactory } from '../common/extractWeightsFactory';
import type { ExtractWeightsFunction, ParamMapping } from '../common/types';
import type { TinyYolov2Config } from './config';
import type { BatchNorm, ConvWithBatchNorm, TinyYolov2NetParams } from './types';

function extractorsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {
  const extractConvParams = extractConvParamsFactory(extractWeights, paramMappings);

  function extractBatchNormParams(size: number, mappedPrefix: string): BatchNorm {
    const sub = tf.tensor1d(extractWeights(size));
    const truediv = tf.tensor1d(extractWeights(size));

    paramMappings.push(
      { paramPath: `${mappedPrefix}/sub` },
      { paramPath: `${mappedPrefix}/truediv` },
    );
    return { sub, truediv };
  }

  function extractConvWithBatchNormParams(channelsIn: number, channelsOut: number, mappedPrefix: string): ConvWithBatchNorm {
    const conv = extractConvParams(channelsIn, channelsOut, 3, `${mappedPrefix}/conv`);
    const bn = extractBatchNormParams(channelsOut, `${mappedPrefix}/bn`);
    return { conv, bn };
  }
  const extractSeparableConvParams = extractSeparableConvParamsFactory(extractWeights, paramMappings);

  return {
    extractConvParams,
    extractConvWithBatchNormParams,
    extractSeparableConvParams,
  };
}

export function extractParams(
  weights: Float32Array,
  config: TinyYolov2Config,
  boxEncodingSize: number,
  filterSizes: number[],
): { params: TinyYolov2NetParams, paramMappings: ParamMapping[] } {
  const {
    extractWeights,
    getRemainingWeights,
  } = extractWeightsFactory(weights);

  const paramMappings: ParamMapping[] = [];
  const {
    extractConvParams,
    extractConvWithBatchNormParams,
    extractSeparableConvParams,
  } = extractorsFactory(extractWeights, paramMappings);
  let params: TinyYolov2NetParams;

  // Validate filterSizes has required elements
  const s0 = filterSizes[0];
  const s1 = filterSizes[1];
  const s2 = filterSizes[2];
  const s3 = filterSizes[3];
  const s4 = filterSizes[4];
  const s5 = filterSizes[5];
  const s6 = filterSizes[6];
  const s7 = filterSizes[7];
  const s8 = filterSizes[8];

  if (s0 === undefined || s1 === undefined || s2 === undefined || s3 === undefined || s4 === undefined || s5 === undefined || s6 === undefined) {
    throw new Error('extractParams - filterSizes must have at least 7 elements');
  }

  if (config.withSeparableConvs) {
    const conv0 = config.isFirstLayerConv2d
      ? extractConvParams(s0, s1, 3, 'conv0')
      : extractSeparableConvParams(s0, s1, 'conv0');
    const conv1 = extractSeparableConvParams(s1, s2, 'conv1');
    const conv2 = extractSeparableConvParams(s2, s3, 'conv2');
    const conv3 = extractSeparableConvParams(s3, s4, 'conv3');
    const conv4 = extractSeparableConvParams(s4, s5, 'conv4');
    const conv5 = extractSeparableConvParams(s5, s6, 'conv5');
    const conv6 = s7 !== undefined ? extractSeparableConvParams(s6, s7, 'conv6') : undefined;
    const conv7 = s7 !== undefined && s8 !== undefined ? extractSeparableConvParams(s7, s8, 'conv7') : undefined;
    const conv8Size = s8 ?? s7 ?? s6;
    const conv8 = extractConvParams(conv8Size, 5 * boxEncodingSize, 1, 'conv8');
    params = {
      conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
    };
  } else {
    if (s7 === undefined || s8 === undefined) {
      throw new Error('extractParams - filterSizes must have 9 elements for non-separable convs');
    }
    const conv0 = extractConvWithBatchNormParams(s0, s1, 'conv0');
    const conv1 = extractConvWithBatchNormParams(s1, s2, 'conv1');
    const conv2 = extractConvWithBatchNormParams(s2, s3, 'conv2');
    const conv3 = extractConvWithBatchNormParams(s3, s4, 'conv3');
    const conv4 = extractConvWithBatchNormParams(s4, s5, 'conv4');
    const conv5 = extractConvWithBatchNormParams(s5, s6, 'conv5');
    const conv6 = extractConvWithBatchNormParams(s6, s7, 'conv6');
    const conv7 = extractConvWithBatchNormParams(s7, s8, 'conv7');
    const conv8 = extractConvParams(s8, 5 * boxEncodingSize, 1, 'conv8');
    params = {
      conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
    };
  }
  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`);
  }
  return { params, paramMappings };
}

import * as tf from '@tensorflow/tfjs';

import { conv, convDown, convNoRelu } from './convLayer';
import { ResidualLayerParams } from './types';

export function residual(x: tf.Tensor4D, params: ResidualLayerParams): tf.Tensor4D {
  return tf.tidy(() => {
    let out = conv(x, params.conv1);
    out = convNoRelu(out, params.conv2);
    out = tf.add(out, x);
    out = tf.relu(out);
    return out;
  });
}

export function residualDown(x: tf.Tensor4D, params: ResidualLayerParams): tf.Tensor4D {
  return tf.tidy(() => {
    let out = convDown(x, params.conv1);
    out = convNoRelu(out, params.conv2);

    let pooled = tf.avgPool(x, 2, 2, 'valid') as tf.Tensor4D;
    const isPad = pooled.shape[3] !== out.shape[3];
    const isAdjustShape = pooled.shape[1] !== out.shape[1] || pooled.shape[2] !== out.shape[2];

    if (isAdjustShape) {
      // Use tf.pad instead of creating separate zero tensors and concatenating
      const padHeight = out.shape[1] < pooled.shape[1] ? 0 : 1;
      const padWidth = out.shape[2] < pooled.shape[2] ? 0 : 1;
      out = tf.pad(out, [[0, 0], [0, padHeight], [0, padWidth], [0, 0]]);
    }

    if (isPad) {
      // Pad channels to match output
      const channelDiff = out.shape[3] - pooled.shape[3];
      pooled = tf.pad(pooled, [[0, 0], [0, 0], [0, 0], [0, channelDiff]]);
    }

    out = tf.add(pooled, out) as tf.Tensor4D;
    out = tf.relu(out);
    return out;
  });
}

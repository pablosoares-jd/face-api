import type * as tf from '@tensorflow/tfjs';

export function seperateWeightMaps(weightMap: tf.NamedTensorMap) {
  const featureExtractorMap: tf.NamedTensorMap = {};
  const classifierMap: tf.NamedTensorMap = {};

  Object.keys(weightMap).forEach((key) => {
    const tensor = weightMap[key];
    if (tensor) {
      const map = key.startsWith('fc') ? classifierMap : featureExtractorMap;
      map[key] = tensor;
    }
  });

  return { featureExtractorMap, classifierMap };
}

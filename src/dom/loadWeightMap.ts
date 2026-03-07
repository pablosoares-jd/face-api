import * as tf from '@tensorflow/tfjs';

import { getModelUris } from '../common/getModelUris';
import { fetchJson } from './fetchJson';

export async function loadWeightMap(
  uri: string | undefined,
  defaultModelName: string,
): Promise<tf.NamedTensorMap> {
  const { manifestUri, modelBaseUri } = getModelUris(uri, defaultModelName);
  // Fetch manifest - it could be a WeightsManifestConfig or wrapped in an object
  const manifestResponse = await fetchJson<tf.io.WeightsManifestConfig | { weightsManifest: tf.io.WeightsManifestConfig }>(manifestUri);

  // Handle both direct manifest format and wrapped format
  const manifest: tf.io.WeightsManifestConfig = 'weightsManifest' in manifestResponse
    ? manifestResponse.weightsManifest
    : manifestResponse;

  // Access loadWeights through io namespace (internal TensorFlow.js API)
  const tfIO = tf.io as typeof tf.io & { loadWeights: (manifest: tf.io.WeightsManifestConfig, baseUri: string) => Promise<tf.NamedTensorMap> };
  return tfIO.loadWeights(manifest, modelBaseUri);
}

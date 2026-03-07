import * as tf from '@tensorflow/tfjs';

import { env } from '../env/index';
import { isTensor4D } from '../utils/index';

export async function imageTensorToCanvas(
  imgTensor: tf.Tensor,
  canvas?: HTMLCanvasElement,
): Promise<HTMLCanvasElement> {
  const targetCanvas = canvas || env.getEnv().createCanvasElement();

  // Validate tensor shape
  const shape = imgTensor.shape;
  const is4D = isTensor4D(imgTensor);
  const sliceStart = is4D ? 1 : 0;

  if (shape.length < sliceStart + 3) {
    throw new Error(`imageTensorToCanvas - expected tensor with at least 3 dimensions, got shape: [${shape.join(', ')}]`);
  }

  const [height, width, numChannels] = shape.slice(sliceStart) as [number, number, number];

  if (!height || !width || !numChannels) {
    throw new Error(`imageTensorToCanvas - invalid tensor dimensions: height=${height}, width=${width}, channels=${numChannels}`);
  }

  const imgTensor3D = tf.tidy(() => imgTensor.as3D(height, width, numChannels).toInt());

  try {
    await tf.browser.toPixels(imgTensor3D, targetCanvas);
    return targetCanvas;
  } finally {
    // Always dispose the tensor, even if toPixels fails
    imgTensor3D.dispose();
  }
}

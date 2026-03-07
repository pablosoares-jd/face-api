import { env } from '../env/index';

export function isMediaElement(input: unknown): input is HTMLImageElement | HTMLCanvasElement | HTMLVideoElement {
  const { Image, Canvas, Video } = env.getEnv();

  return input instanceof Image
    || input instanceof Canvas
    || input instanceof Video;
}

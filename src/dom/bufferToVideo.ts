import { env } from '../env/index';

export function bufferToVideo(buf: Blob): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    if (!(buf instanceof Blob)) {
      reject(new Error('bufferToVideo - expected buf to be of type: Blob'));
      return;
    }

    const video = env.getEnv().createVideoElement();
    const blobUrl = URL.createObjectURL(buf);

    const cleanup = () => {
      // Revoke the blob URL to free memory
      URL.revokeObjectURL(blobUrl);
      // Remove event listeners to prevent memory leaks
      video.oncanplay = null;
      video.onerror = null;
      video.onabort = null;
    };

    video.oncanplay = () => {
      cleanup();
      resolve(video);
    };

    video.onerror = (err) => {
      cleanup();
      reject(err);
    };

    video.onabort = () => {
      cleanup();
      reject(new Error('bufferToVideo - video loading was aborted'));
    };

    video.playsInline = true;
    video.muted = true;
    video.src = blobUrl;
    video.play();
  });
}

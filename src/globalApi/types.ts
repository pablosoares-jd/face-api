import type { FaceDetection } from '../classes/FaceDetection';
import type { TNetInput } from '../dom/index';
import type { SsdMobilenetv1Options } from '../ssdMobilenetv1/SsdMobilenetv1Options';
import type { TinyFaceDetectorOptions } from '../tinyFaceDetector/TinyFaceDetectorOptions';
import type { TinyYolov2Options } from '../tinyYolov2/index';

export type FaceDetectionOptions = TinyFaceDetectorOptions | SsdMobilenetv1Options | TinyYolov2Options

// eslint-disable-next-line no-unused-vars
export type FaceDetectionFunction = (input: TNetInput) => Promise<FaceDetection[]>

import type { ITinyYolov2Options } from '../tinyYolov2/index';
import { TinyYolov2Options } from '../tinyYolov2/index';

export type ITinyFaceDetectorOptions = ITinyYolov2Options

export class TinyFaceDetectorOptions extends TinyYolov2Options {
  protected override _name = 'TinyFaceDetectorOptions';
}

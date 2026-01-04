/* eslint-disable max-classes-per-file */
import { FaceDetection } from '../classes/FaceDetection';
import { TNetInput } from '../dom/index';
import { extendWithFaceDetection, WithFaceDetection } from '../factories/WithFaceDetection';
import { SsdMobilenetv1Options } from '../ssdMobilenetv1/SsdMobilenetv1Options';
import { TinyFaceDetectorOptions } from '../tinyFaceDetector/TinyFaceDetectorOptions';
import { TinyYolov2Options } from '../tinyYolov2/index';
import { ComposableTask } from './ComposableTask';
import { DetectAllFaceLandmarksTask, DetectSingleFaceLandmarksTask } from './DetectFaceLandmarksTasks';
import { nets } from './nets';
import { PredictAllAgeAndGenderTask, PredictSingleAgeAndGenderTask } from './PredictAgeAndGenderTask';
import { PredictAllFaceExpressionsTask, PredictSingleFaceExpressionsTask } from './PredictFaceExpressionsTask';
import { FaceDetectionOptions } from './types';

export class DetectFacesTaskBase<TReturn> extends ComposableTask<TReturn> {
  // eslint-disable-next-line no-unused-vars
  constructor(protected input: TNetInput, protected options: FaceDetectionOptions = new SsdMobilenetv1Options()) {
    super();
  }
}

export class DetectAllFacesTask extends DetectFacesTaskBase<FaceDetection[]> {
  public override async run(): Promise<FaceDetection[]> {
    const { input, options } = this;
    let result;
    if (options instanceof TinyFaceDetectorOptions) result = nets.tinyFaceDetector.locateFaces(input, options);
    else if (options instanceof SsdMobilenetv1Options) result = nets.ssdMobilenetv1.locateFaces(input, options);
    else if (options instanceof TinyYolov2Options) result = nets.tinyYolov2.locateFaces(input, options);
    else throw new Error('detectFaces - expected options to be instance of TinyFaceDetectorOptions | SsdMobilenetv1Options | TinyYolov2Options');
    return result;
  }

  private async runAndExtendWithFaceDetections(): Promise<WithFaceDetection<{}>[]> {
    const detections = await this.run();
    return detections.map((detection) => extendWithFaceDetection({}, detection));
  }

  withFaceLandmarks(useTinyLandmarkNet = false) {
    return new DetectAllFaceLandmarksTask(
      this.runAndExtendWithFaceDetections(),
      this.input,
      useTinyLandmarkNet,
    );
  }

  withFaceExpressions() {
    return new PredictAllFaceExpressionsTask(
      this.runAndExtendWithFaceDetections(),
      this.input,
    );
  }

  withAgeAndGender() {
    return new PredictAllAgeAndGenderTask(
      this.runAndExtendWithFaceDetections(),
      this.input,
    );
  }
}

export class DetectSingleFaceTask extends DetectFacesTaskBase<FaceDetection | undefined> {
  public override async run(): Promise<FaceDetection | undefined> {
    const faceDetections = await new DetectAllFacesTask(this.input, this.options).run();

    // Handle empty detections array - return undefined instead of crashing
    if (!faceDetections || faceDetections.length === 0) {
      return undefined;
    }

    // Find face with highest score using reduce for better performance
    return faceDetections.reduce((best, current) =>
      current.score > best.score ? current : best
    );
  }

  private async runAndExtendWithFaceDetection(): Promise<WithFaceDetection<{}> | undefined> {
    const detection = await this.run();
    return detection ? extendWithFaceDetection<{}>({}, detection) : undefined;
  }

  withFaceLandmarks(useTinyLandmarkNet = false) {
    return new DetectSingleFaceLandmarksTask(
      this.runAndExtendWithFaceDetection(),
      this.input,
      useTinyLandmarkNet,
    );
  }

  withFaceExpressions() {
    return new PredictSingleFaceExpressionsTask(
      this.runAndExtendWithFaceDetection(),
      this.input,
    );
  }

  withAgeAndGender() {
    return new PredictSingleAgeAndGenderTask(
      this.runAndExtendWithFaceDetection(),
      this.input,
    );
  }
}

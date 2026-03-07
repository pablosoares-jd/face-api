/* eslint-disable max-classes-per-file */
import * as tf from '@tensorflow/tfjs';

import type { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import type { TNetInput } from '../dom/index';
import { extractFaces, extractFaceTensors } from '../dom/index';
import type { FaceLandmark68Net } from '../faceLandmarkNet/FaceLandmark68Net';
import type { FaceLandmark68TinyNet } from '../faceLandmarkNet/FaceLandmark68TinyNet';
import type { WithFaceDetection } from '../factories/WithFaceDetection';
import type { WithFaceLandmarks } from '../factories/WithFaceLandmarks';
import { extendWithFaceLandmarks } from '../factories/WithFaceLandmarks';
import { ComposableTask } from './ComposableTask';
import {
  ComputeAllFaceDescriptorsTask,
  ComputeSingleFaceDescriptorTask,
} from './ComputeFaceDescriptorsTasks';
import { nets } from './nets';
import {
  PredictAllAgeAndGenderWithFaceAlignmentTask,
  PredictSingleAgeAndGenderWithFaceAlignmentTask,
} from './PredictAgeAndGenderTask';
import {
  PredictAllFaceExpressionsWithFaceAlignmentTask,
  PredictSingleFaceExpressionsWithFaceAlignmentTask,
} from './PredictFaceExpressionsTask';

export class DetectFaceLandmarksTaskBase<TReturn, TParentReturn> extends ComposableTask<TReturn> {
  constructor(
    // eslint-disable-next-line no-unused-vars
    protected parentTask: ComposableTask<TParentReturn> | Promise<TParentReturn>,
    // eslint-disable-next-line no-unused-vars
    protected input: TNetInput,
    // eslint-disable-next-line no-unused-vars
    protected useTinyLandmarkNet: boolean,
  ) {
    super();
  }

  protected get landmarkNet(): FaceLandmark68Net | FaceLandmark68TinyNet {
    return this.useTinyLandmarkNet
      ? nets.faceLandmark68TinyNet
      : nets.faceLandmark68Net;
  }
}

export class DetectAllFaceLandmarksTask<TSource extends WithFaceDetection<{}>> extends DetectFaceLandmarksTaskBase<WithFaceLandmarks<TSource>[], TSource[]> {
  public override async run(): Promise<WithFaceLandmarks<TSource>[]> {
    const parentResults = await this.parentTask;
    const detections = parentResults.map((res) => res.detection);
    const faces: Array<HTMLCanvasElement | tf.Tensor3D> = this.input instanceof tf.Tensor
      ? await extractFaceTensors(this.input, detections)
      : await extractFaces(this.input, detections);
    const faceLandmarksByFace = await Promise.all(faces.map((face) => this.landmarkNet.detectLandmarks(face))) as FaceLandmarks68[];
    faces.forEach((f) => f instanceof tf.Tensor && f.dispose());

    // Use flatMap to filter and map while preserving correct index alignment
    // This avoids the bug where filter changes indices but map still uses original indices
    const result = parentResults.flatMap((parentResult, i) => {
      const landmarks = faceLandmarksByFace[i];
      if (!landmarks) {
        return []; // Skip this result (equivalent to filter returning false)
      }
      return [extendWithFaceLandmarks<TSource>(parentResult, landmarks)];
    });

    return result;
  }

  withFaceExpressions() {
    return new PredictAllFaceExpressionsWithFaceAlignmentTask(this, this.input);
  }

  withAgeAndGender() {
    return new PredictAllAgeAndGenderWithFaceAlignmentTask(this, this.input);
  }

  withFaceDescriptors() {
    return new ComputeAllFaceDescriptorsTask(this, this.input);
  }
}

export class DetectSingleFaceLandmarksTask<
  TSource extends WithFaceDetection<{}>,
> extends DetectFaceLandmarksTaskBase<WithFaceLandmarks<TSource> | undefined, TSource | undefined> {
  public override async run(): Promise<WithFaceLandmarks<TSource> | undefined> {
    const parentResult = await this.parentTask;
    if (!parentResult) {
      return undefined;
    }
    const { detection } = parentResult;
    const faces: Array<HTMLCanvasElement | tf.Tensor3D> = this.input instanceof tf.Tensor
      ? await extractFaceTensors(this.input, [detection])
      : await extractFaces(this.input, [detection]);
    const face = faces[0];
    if (!face) {
      faces.forEach((f) => f instanceof tf.Tensor && f.dispose());
      return undefined;
    }
    const landmarks = await this.landmarkNet.detectLandmarks(face) as FaceLandmarks68;
    faces.forEach((f) => f instanceof tf.Tensor && f.dispose());
    return extendWithFaceLandmarks<TSource>(parentResult, landmarks);
  }

  withFaceExpressions() {
    return new PredictSingleFaceExpressionsWithFaceAlignmentTask(this, this.input);
  }

  withAgeAndGender() {
    return new PredictSingleAgeAndGenderWithFaceAlignmentTask(this, this.input);
  }

  withFaceDescriptor() {
    return new ComputeSingleFaceDescriptorTask(this, this.input);
  }
}

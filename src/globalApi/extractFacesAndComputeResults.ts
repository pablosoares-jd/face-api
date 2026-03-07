import * as tf from '@tensorflow/tfjs';

import type { FaceDetection } from '../classes/FaceDetection';
import type { TNetInput } from '../dom/index';
import { extractFaces, extractFaceTensors } from '../dom/index';
import type { WithFaceDetection } from '../factories/WithFaceDetection';
import type { WithFaceLandmarks } from '../factories/WithFaceLandmarks';
import { isWithFaceLandmarks } from '../factories/WithFaceLandmarks';

export async function extractAllFacesAndComputeResults<TSource extends WithFaceDetection<{}>, TResult>(
  parentResults: TSource[],
  input: TNetInput,
  // eslint-disable-next-line no-unused-vars
  computeResults: (faces: Array<HTMLCanvasElement | tf.Tensor3D>) => Promise<TResult>,
  extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | null,
  // eslint-disable-next-line no-unused-vars
  getRectForAlignment: (parentResult: WithFaceLandmarks<TSource, any>) => FaceDetection = ({ alignedRect }) => alignedRect,
) {
  const faceBoxes = parentResults.map((parentResult) => (isWithFaceLandmarks(parentResult)
    ? getRectForAlignment(parentResult)
    : parentResult.detection));
  const faces: Array<HTMLCanvasElement | tf.Tensor3D> = extractedFaces || (
    input instanceof tf.Tensor
      ? await extractFaceTensors(input, faceBoxes)
      : await extractFaces(input, faceBoxes)
  );
  const results = await computeResults(faces);
  faces.forEach((f) => f instanceof tf.Tensor && f.dispose());
  return results;
}

export async function extractSingleFaceAndComputeResult<TSource extends WithFaceDetection<{}>, TResult>(
  parentResult: TSource,
  input: TNetInput,
  // eslint-disable-next-line no-unused-vars
  computeResult: (face: HTMLCanvasElement | tf.Tensor3D) => Promise<TResult>,
  extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | null,
  // eslint-disable-next-line no-unused-vars
  getRectForAlignment?: (parentResultLocal: WithFaceLandmarks<TSource, any>) => FaceDetection,
) {
  return extractAllFacesAndComputeResults<TSource, TResult>(
    [parentResult],
    input,
    async (faces) => {
      const face = faces[0];
      if (!face) {
        throw new Error('extractSingleFaceAndComputeResult - no face extracted');
      }
      return computeResult(face);
    },
    extractedFaces,
    getRectForAlignment,
  );
}

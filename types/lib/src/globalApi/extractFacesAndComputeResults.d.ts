import * as tf from '@tensorflow/tfjs';
import type { FaceDetection } from '../classes/FaceDetection';
import type { TNetInput } from '../dom/index';
import type { WithFaceDetection } from '../factories/WithFaceDetection';
import type { WithFaceLandmarks } from '../factories/WithFaceLandmarks';
export declare function extractAllFacesAndComputeResults<TSource extends WithFaceDetection<{}>, TResult>(parentResults: TSource[], input: TNetInput, computeResults: (faces: Array<HTMLCanvasElement | tf.Tensor3D>) => Promise<TResult>, extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | null, getRectForAlignment?: (parentResult: WithFaceLandmarks<TSource, any>) => FaceDetection): Promise<TResult>;
export declare function extractSingleFaceAndComputeResult<TSource extends WithFaceDetection<{}>, TResult>(parentResult: TSource, input: TNetInput, computeResult: (face: HTMLCanvasElement | tf.Tensor3D) => Promise<TResult>, extractedFaces?: Array<HTMLCanvasElement | tf.Tensor3D> | null, getRectForAlignment?: (parentResultLocal: WithFaceLandmarks<TSource, any>) => FaceDetection): Promise<TResult>;

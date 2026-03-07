import type { IBoundingBox, IRect } from '../classes/index';
import { FaceDetection } from '../classes/FaceDetection';
import type { WithFaceDetection } from '../factories/WithFaceDetection';
export type TDrawDetectionsInput = IRect | IBoundingBox | FaceDetection | WithFaceDetection<{}>;
export declare function drawDetections(canvasArg: string | HTMLCanvasElement, detections: TDrawDetectionsInput | Array<TDrawDetectionsInput>): void;

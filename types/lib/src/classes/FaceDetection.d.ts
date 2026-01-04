import { Box } from './Box';
import { IDimensions } from './Dimensions';
import { ObjectDetection } from './ObjectDetection';
import { Rect } from './Rect';
export interface IFaceDetection {
    score: number;
    box: Box;
}
export declare class FaceDetection extends ObjectDetection implements IFaceDetection {
    constructor(score: number, relativeBox: Rect, imageDims: IDimensions);
    forSize(width: number, height: number): FaceDetection;
}

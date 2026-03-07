import type * as tf from '@tensorflow/tfjs';
import type { Point } from '../classes/index';
import { FaceDetection } from '../classes/index';
import type { ParamMapping } from '../common/index';
import type { TNetInput } from '../dom/index';
import type { ITinyYolov2Options } from '../tinyYolov2/index';
import { TinyYolov2Base } from '../tinyYolov2/TinyYolov2Base';
import type { TinyYolov2NetParams } from '../tinyYolov2/types';
export declare class TinyFaceDetector extends TinyYolov2Base {
    constructor();
    get anchors(): Point[];
    locateFaces(input: TNetInput, forwardParams: ITinyYolov2Options): Promise<FaceDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TinyYolov2NetParams;
        paramMappings: ParamMapping[];
    };
}

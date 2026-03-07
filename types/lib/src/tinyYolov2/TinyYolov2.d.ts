import type * as tf from '@tensorflow/tfjs';
import type { Point } from '../classes/index';
import { FaceDetection } from '../classes/index';
import type { ParamMapping } from '../common/types';
import type { TNetInput } from '../dom/types';
import { TinyYolov2Base } from './TinyYolov2Base';
import type { ITinyYolov2Options } from './TinyYolov2Options';
import type { TinyYolov2NetParams } from './types';
export declare class TinyYolov2 extends TinyYolov2Base {
    constructor(withSeparableConvs?: boolean);
    get withSeparableConvs(): boolean;
    get anchors(): Point[];
    locateFaces(input: TNetInput, forwardParams: ITinyYolov2Options): Promise<FaceDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeightMap(weightMap: tf.NamedTensorMap): {
        params: TinyYolov2NetParams;
        paramMappings: ParamMapping[];
    };
}

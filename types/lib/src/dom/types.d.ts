import type { Tensor3D, Tensor4D } from '@tensorflow/tfjs';
import { NetInput } from './NetInput';
export type TMediaElement = HTMLImageElement | HTMLVideoElement | HTMLCanvasElement;
export type TResolvedNetInput = TMediaElement | Tensor3D | Tensor4D;
export type TNetInput = string | TResolvedNetInput | Array<string | TResolvedNetInput> | NetInput;

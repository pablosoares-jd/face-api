import type { IBoundingBox } from './BoundingBox';
import { Box } from './Box';
import type { IRect } from './Rect';
export declare class LabeledBox extends Box {
    static assertIsValidLabeledBox(box: any, callee: string): void;
    private _label;
    constructor(box: IBoundingBox | IRect | any, label: number);
    get label(): number;
}

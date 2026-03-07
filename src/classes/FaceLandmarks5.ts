import { getCenterPoint } from '../utils/index';
import { FaceLandmarks } from './FaceLandmarks';
import type { Point } from './Point';

export class FaceLandmarks5 extends FaceLandmarks {
  protected override getRefPointsForAlignment(): Point[] {
    const pts = this.positions;
    const pt0 = pts[0];
    const pt1 = pts[1];
    const pt3 = pts[3];
    const pt4 = pts[4];

    if (!pt0 || !pt1 || !pt3 || !pt4) {
      throw new Error('FaceLandmarks5.getRefPointsForAlignment - insufficient landmarks');
    }

    return [
      pt0,
      pt1,
      getCenterPoint([pt3, pt4]),
    ];
  }
}

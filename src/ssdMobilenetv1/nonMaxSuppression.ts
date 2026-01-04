import * as tf from '../../dist/tfjs.esm';

/**
 * Calculate IOU (Intersection over Union) between two boxes using pre-fetched data.
 * Avoids GPU blocking by using already-extracted array data.
 */
function calculateIOU(boxesData: number[][], i: number, j: number): number {
  const boxI = boxesData[i];
  const boxJ = boxesData[j];

  const yminI = Math.min(boxI[0], boxI[2]);
  const xminI = Math.min(boxI[1], boxI[3]);
  const ymaxI = Math.max(boxI[0], boxI[2]);
  const xmaxI = Math.max(boxI[1], boxI[3]);

  const yminJ = Math.min(boxJ[0], boxJ[2]);
  const xminJ = Math.min(boxJ[1], boxJ[3]);
  const ymaxJ = Math.max(boxJ[0], boxJ[2]);
  const xmaxJ = Math.max(boxJ[1], boxJ[3]);

  const areaI = (ymaxI - yminI) * (xmaxI - xminI);
  const areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);

  if (areaI <= 0 || areaJ <= 0) return 0.0;

  const intersectionYmin = Math.max(yminI, yminJ);
  const intersectionXmin = Math.max(xminI, xminJ);
  const intersectionYmax = Math.min(ymaxI, ymaxJ);
  const intersectionXmax = Math.min(xmaxI, xmaxJ);

  const intersectionArea =
    Math.max(intersectionYmax - intersectionYmin, 0.0) *
    Math.max(intersectionXmax - intersectionXmin, 0.0);

  return intersectionArea / (areaI + areaJ - intersectionArea);
}

/**
 * Non-Maximum Suppression for SSD MobileNet.
 * Uses pre-fetched box data to avoid GPU blocking during IOU calculations.
 *
 * Note: The boxes tensor data should be fetched before calling this function
 * using async boxes.array() for better performance.
 */
export function nonMaxSuppression(
  boxes: tf.Tensor2D | number[][],
  scores: number[],
  maxOutputSize: number,
  iouThreshold: number,
  scoreThreshold: number,
): number[] {
  // If boxes is a tensor, extract data synchronously (caller should pre-fetch for perf)
  // This maintains backward compatibility while allowing optimized usage
  const boxesData: number[][] = Array.isArray(boxes)
    ? boxes
    : (boxes as tf.Tensor2D).arraySync() as number[][];

  const numBoxes = boxesData.length;
  const outputSize = Math.min(maxOutputSize, numBoxes);

  // Filter and sort candidates by score (descending)
  const candidates = scores
    .map((score, boxIndex) => ({ score, boxIndex }))
    .filter((c) => c.score > scoreThreshold)
    .sort((c1, c2) => c2.score - c1.score);

  const suppressFunc = (iou: number) => (iou <= iouThreshold ? 1 : 0);
  const selected: number[] = [];

  for (const c of candidates) {
    if (selected.length >= outputSize) break;

    const originalScore = c.score;

    // Check IOU against all previously selected boxes
    for (let j = selected.length - 1; j >= 0; --j) {
      const iou = calculateIOU(boxesData, c.boxIndex, selected[j]);
      if (iou === 0.0) continue;

      c.score *= suppressFunc(iou);
      if (c.score <= scoreThreshold) break;
    }

    // Only select if score wasn't suppressed
    if (originalScore === c.score) {
      selected.push(c.boxIndex);
    }
  }

  return selected;
}

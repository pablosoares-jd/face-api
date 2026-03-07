export function euclideanDistance(arr1: number[] | Float32Array | null | undefined, arr2: number[] | Float32Array | null | undefined): number {
  // Defensive null checks
  if (!arr1 || !arr2) {
    throw new Error('euclideanDistance: arr1 and arr2 must not be null or undefined');
  }
  if (arr1.length !== arr2.length) {
    throw new Error(`euclideanDistance: arr1.length (${arr1.length}) !== arr2.length (${arr2.length})`);
  }
  const desc1 = Array.from(arr1);
  const desc2 = Array.from(arr2);
  return Math.sqrt(
    desc1
      .map((val, i) => val - (desc2[i] ?? 0))
      .reduce((res, diff) => res + (diff * diff), 0),
  );
}

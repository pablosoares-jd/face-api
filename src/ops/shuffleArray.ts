export function shuffleArray<T>(inputArray: T[]): T[] {
  const array = inputArray.slice();
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const x = array[i] as T;
    array[i] = array[j] as T;
    array[j] = x;
  }
  return array;
}

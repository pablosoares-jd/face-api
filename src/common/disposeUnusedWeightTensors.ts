import type { ParamMapping } from './types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function disposeUnusedWeightTensors(weightMap: any, paramMappings: ParamMapping[]) {
  Object.keys(weightMap).forEach((path) => {
    if (!paramMappings.some((pm) => pm.originalPath === path)) {
      weightMap[path].dispose();
    }
  });
}

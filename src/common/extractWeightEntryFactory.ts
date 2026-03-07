import { isTensor } from '../utils/index';
import type { ParamMapping } from './types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function extractWeightEntryFactory(weightMap: any, paramMappings: ParamMapping[]) {
  return (originalPath: string, paramRank: number, mappedPath?: string) => {
    const tensor = weightMap[originalPath];

    if (!isTensor(tensor, paramRank)) {
      throw new Error(`expected weightMap[${originalPath}] to be a Tensor${paramRank}D, instead have ${tensor}`);
    }

    paramMappings.push(
      { originalPath, paramPath: mappedPath || originalPath },
    );

    return tensor;
  };
}

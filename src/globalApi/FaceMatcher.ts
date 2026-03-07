import { FaceMatch } from '../classes/FaceMatch';
import { LabeledFaceDescriptors } from '../classes/LabeledFaceDescriptors';
import { euclideanDistance } from '../euclideanDistance';
import type { WithFaceDescriptor } from '../factories/index';

export class FaceMatcher {
  private _labeledDescriptors: LabeledFaceDescriptors[];
  private _distanceThreshold: number;

  constructor(
    inputs: LabeledFaceDescriptors | WithFaceDescriptor<unknown> | Float32Array
      | Array<LabeledFaceDescriptors | WithFaceDescriptor<unknown> | Float32Array>,
    distanceThreshold = 0.6,
  ) {
    this._distanceThreshold = distanceThreshold;
    const inputArray = Array.isArray(inputs) ? inputs : [inputs];
    if (!inputArray.length) throw new Error('FaceRecognizer.constructor - expected atleast one input');
    let count = 1;
    const createUniqueLabel = () => `person ${count++}`;
    this._labeledDescriptors = inputArray.map((desc) => {
      if (desc instanceof LabeledFaceDescriptors) return desc;
      if (desc instanceof Float32Array) return new LabeledFaceDescriptors(createUniqueLabel(), [desc]);
      if (desc.descriptor && desc.descriptor instanceof Float32Array) return new LabeledFaceDescriptors(createUniqueLabel(), [desc.descriptor]);
      throw new Error('FaceRecognizer.constructor - expected inputs to be of type LabeledFaceDescriptors | WithFaceDescriptor<any> | Float32Array | Array<LabeledFaceDescriptors | WithFaceDescriptor<any> | Float32Array>');
    });
  }

  public get labeledDescriptors(): LabeledFaceDescriptors[] { return this._labeledDescriptors; }

  public get distanceThreshold(): number { return this._distanceThreshold; }

  public computeMeanDistance(queryDescriptor: Float32Array, descriptors: Float32Array[]): number {
    if (!descriptors || descriptors.length === 0) {
      return Infinity; // No descriptors to compare against
    }

    const totalDistance = descriptors
      .map((d) => euclideanDistance(d, queryDescriptor))
      .reduce((d1, d2) => d1 + d2, 0);

    return totalDistance / descriptors.length;
  }

  public matchDescriptor(queryDescriptor: Float32Array): FaceMatch {
    if (!this.labeledDescriptors || this.labeledDescriptors.length === 0) {
      throw new Error('FaceMatcher.matchDescriptor - no labeled descriptors available. Add descriptors before matching.');
    }

    const matches = this.labeledDescriptors
      .map(({ descriptors, label }) => new FaceMatch(label, this.computeMeanDistance(queryDescriptor, descriptors)));

    // Find best match (lowest distance)
    return matches.reduce((best, curr) => (best.distance < curr.distance ? best : curr));
  }

  public findBestMatch(queryDescriptor: Float32Array): FaceMatch {
    const bestMatch = this.matchDescriptor(queryDescriptor);
    return (bestMatch.distance < this._distanceThreshold) ? bestMatch : new FaceMatch('unknown', bestMatch.distance);
  }

  public toJSON(): any {
    return {
      distanceThreshold: this._distanceThreshold,
      labeledDescriptors: this._labeledDescriptors.map((ld) => ld.toJSON()),
    };
  }

  public static fromJSON(json: any): FaceMatcher {
    const labeledDescriptors = json.labeledDescriptors.map((ld: any) => LabeledFaceDescriptors.fromJSON(ld));
    return new FaceMatcher(labeledDescriptors, json.distanceThreshold);
  }
}

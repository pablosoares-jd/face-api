/**
 * KNN-based Face Classifier for real-time face recognition.
 *
 * Uses face descriptors (128-dim embeddings) to classify faces
 * without requiring gradient-based training.
 *
 * @example
 * ```typescript
 * import { FaceClassifier } from '@vladmandic/face-api/training';
 * import * as faceapi from '@vladmandic/face-api';
 *
 * const classifier = new FaceClassifier();
 *
 * // Add known faces
 * const descriptor = await faceapi.computeFaceDescriptor(image);
 * classifier.addFace(descriptor, 'Pablo');
 * classifier.addFace(descriptor2, 'Maria');
 *
 * // Recognize new faces
 * const result = classifier.classify(newDescriptor);
 * console.log(`${result.label} (${result.confidence}%)`);
 *
 * // Save/load classifier
 * const json = classifier.toJSON();
 * classifier.fromJSON(json);
 * ```
 */

import * as tf from '@tensorflow/tfjs';
import { euclideanDistance } from '../euclideanDistance';

/**
 * A labeled face descriptor.
 */
export interface LabeledFaceDescriptor {
  label: string;
  descriptor: Float32Array;
}

/**
 * Classification result.
 */
export interface ClassificationResult {
  /** Best matching label */
  label: string;
  /** Distance to closest match (lower = better) */
  distance: number;
  /** Confidence score (0-1, higher = better) */
  confidence: number;
  /** All matches sorted by distance */
  matches: Array<{
    label: string;
    distance: number;
    confidence: number;
  }>;
}

/**
 * Options for classification.
 */
export interface ClassifyOptions {
  /** Maximum distance threshold for a valid match (default: 0.6) */
  threshold?: number;
  /** Number of nearest neighbors to consider (default: 1) */
  k?: number;
}

/**
 * Serialized classifier state.
 */
export interface SerializedClassifier {
  version: number;
  descriptorSize: number;
  faces: Array<{
    label: string;
    descriptor: number[];
  }>;
  metadata?: Record<string, unknown>;
}

/**
 * KNN-based Face Classifier.
 *
 * Features:
 * - Add faces incrementally without retraining
 * - K-nearest neighbors classification
 * - Confidence scoring
 * - Save/load classifier state
 * - Memory efficient (stores only 128 floats per face)
 */
export class FaceClassifier {
  private _faces: LabeledFaceDescriptor[] = [];
  private _descriptorSize = 128;
  private _metadata: Record<string, unknown> = {};

  constructor() {}

  /**
   * Add a face descriptor with a label.
   *
   * @param descriptor - 128-dimensional face descriptor
   * @param label - Identity label for this face
   */
  public addFace(descriptor: Float32Array | number[], label: string): void {
    const desc = descriptor instanceof Float32Array
      ? descriptor
      : new Float32Array(descriptor);

    if (desc.length !== this._descriptorSize) {
      throw new Error(
        `Invalid descriptor size: expected ${this._descriptorSize}, got ${desc.length}`
      );
    }

    this._faces.push({ label, descriptor: desc });
  }

  /**
   * Add multiple faces for a single identity.
   *
   * @param descriptors - Array of face descriptors
   * @param label - Identity label
   */
  public addFaces(descriptors: Array<Float32Array | number[]>, label: string): void {
    for (const desc of descriptors) {
      this.addFace(desc, label);
    }
  }

  /**
   * Remove all faces for a given label.
   *
   * @param label - Label to remove
   * @returns Number of faces removed
   */
  public removeFaces(label: string): number {
    const initialCount = this._faces.length;
    this._faces = this._faces.filter(f => f.label !== label);
    return initialCount - this._faces.length;
  }

  /**
   * Clear all faces.
   */
  public clear(): void {
    this._faces = [];
  }

  /**
   * Classify a face descriptor.
   *
   * @param descriptor - Face descriptor to classify
   * @param options - Classification options
   * @returns Classification result
   */
  public classify(
    descriptor: Float32Array | number[],
    options: ClassifyOptions = {}
  ): ClassificationResult {
    const { threshold = 0.6, k = 1 } = options;

    if (this._faces.length === 0) {
      return {
        label: 'unknown',
        distance: Infinity,
        confidence: 0,
        matches: []
      };
    }

    const queryDesc = descriptor instanceof Float32Array
      ? Array.from(descriptor)
      : descriptor;

    // Calculate distances to all faces
    const distances = this._faces.map(face => ({
      label: face.label,
      distance: euclideanDistance(queryDesc, Array.from(face.descriptor))
    }));

    // Sort by distance
    distances.sort((a, b) => a.distance - b.distance);

    // Apply KNN voting
    const kNearest = distances.slice(0, k);
    const votes: Record<string, number> = {};

    for (const match of kNearest) {
      votes[match.label] = (votes[match.label] || 0) + 1;
    }

    // Find winner
    let bestLabel = 'unknown';
    let bestVotes = 0;
    for (const [label, count] of Object.entries(votes)) {
      if (count > bestVotes) {
        bestVotes = count;
        bestLabel = label;
      }
    }

    const bestDistance = distances[0].distance;

    // Convert distance to confidence (sigmoid-like curve)
    const confidence = bestDistance < threshold
      ? Math.max(0, 1 - (bestDistance / threshold))
      : 0;

    // Check threshold
    if (bestDistance > threshold) {
      bestLabel = 'unknown';
    }

    return {
      label: bestLabel,
      distance: bestDistance,
      confidence,
      matches: distances.slice(0, 10).map(m => ({
        label: m.label,
        distance: m.distance,
        confidence: Math.max(0, 1 - (m.distance / threshold))
      }))
    };
  }

  /**
   * Classify using TensorFlow.js tensors for batch processing.
   * More efficient for classifying multiple faces at once.
   *
   * @param descriptors - Tensor of shape [N, 128]
   * @param options - Classification options
   */
  public classifyBatch(
    descriptors: tf.Tensor2D,
    options: ClassifyOptions = {}
  ): ClassificationResult[] {
    const descriptorArray = descriptors.arraySync() as number[][];
    return descriptorArray.map(desc => this.classify(desc, options));
  }

  /**
   * Get all unique labels in the classifier.
   */
  public getLabels(): string[] {
    return [...new Set(this._faces.map(f => f.label))];
  }

  /**
   * Get number of faces for each label.
   */
  public getLabelCounts(): Record<string, number> {
    const counts: Record<string, number> = {};
    for (const face of this._faces) {
      counts[face.label] = (counts[face.label] || 0) + 1;
    }
    return counts;
  }

  /**
   * Get total number of faces.
   */
  public get size(): number {
    return this._faces.length;
  }

  /**
   * Check if classifier has any faces.
   */
  public get isEmpty(): boolean {
    return this._faces.length === 0;
  }

  /**
   * Set metadata.
   */
  public setMetadata(key: string, value: unknown): void {
    this._metadata[key] = value;
  }

  /**
   * Get metadata.
   */
  public getMetadata(key: string): unknown {
    return this._metadata[key];
  }

  /**
   * Serialize classifier to JSON.
   */
  public toJSON(): SerializedClassifier {
    return {
      version: 1,
      descriptorSize: this._descriptorSize,
      faces: this._faces.map(f => ({
        label: f.label,
        descriptor: Array.from(f.descriptor)
      })),
      metadata: this._metadata
    };
  }

  /**
   * Load classifier from JSON.
   */
  public fromJSON(data: SerializedClassifier): void {
    if (data.version !== 1) {
      throw new Error(`Unsupported classifier version: ${data.version}`);
    }

    this._descriptorSize = data.descriptorSize;
    this._faces = data.faces.map(f => ({
      label: f.label,
      descriptor: new Float32Array(f.descriptor)
    }));
    this._metadata = data.metadata || {};
  }

  /**
   * Save classifier to a JSON string.
   */
  public save(): string {
    return JSON.stringify(this.toJSON());
  }

  /**
   * Load classifier from a JSON string.
   */
  public load(jsonString: string): void {
    this.fromJSON(JSON.parse(jsonString));
  }

  /**
   * Compute average descriptor for a label (centroid).
   * Useful for creating representative embeddings.
   */
  public computeCentroid(label: string): Float32Array | null {
    const faces = this._faces.filter(f => f.label === label);
    if (faces.length === 0) return null;

    const centroid = new Float32Array(this._descriptorSize);
    for (const face of faces) {
      for (let i = 0; i < this._descriptorSize; i++) {
        centroid[i] += face.descriptor[i];
      }
    }
    for (let i = 0; i < this._descriptorSize; i++) {
      centroid[i] /= faces.length;
    }

    // Normalize to unit length
    let norm = 0;
    for (let i = 0; i < this._descriptorSize; i++) {
      norm += centroid[i] * centroid[i];
    }
    norm = Math.sqrt(norm);
    for (let i = 0; i < this._descriptorSize; i++) {
      centroid[i] /= norm;
    }

    return centroid;
  }

  /**
   * Reduce classifier to centroids only.
   * Useful for reducing memory footprint when many samples per identity.
   */
  public reduceToCentroids(): void {
    const labels = this.getLabels();
    const centroids: LabeledFaceDescriptor[] = [];

    for (const label of labels) {
      const centroid = this.computeCentroid(label);
      if (centroid) {
        centroids.push({ label, descriptor: centroid });
      }
    }

    this._faces = centroids;
  }
}

/**
 * Dataset Loader for face-api training.
 *
 * Supports multiple formats:
 * - Directory with class folders
 * - CSV with image paths and labels
 * - JSON manifest
 * - URL streaming
 *
 * @example
 * ```typescript
 * import { DatasetLoader } from '@vladmandic/face-api/training';
 *
 * const loader = new DatasetLoader();
 *
 * // Load from directory structure
 * const dataset = await loader.fromDirectory('./faces', {
 *   task: 'recognition'
 * });
 *
 * // Load from CSV
 * const dataset = await loader.fromCSV('./labels.csv', {
 *   imageColumn: 'path',
 *   labelColumns: ['age', 'gender']
 * });
 *
 * // Create TensorFlow dataset
 * const tfDataset = dataset.toTensorDataset({ batchSize: 32 });
 * ```
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Training task type.
 */
export type TrainingTask = 'recognition' | 'expression' | 'age-gender' | 'landmarks' | 'detection';

/**
 * A single training sample.
 */
export interface TrainingSample {
  /** Image data (path, URL, base64, or tensor) */
  image: string | ImageData | tf.Tensor3D;
  /** Labels for the sample */
  labels: {
    identity?: string;
    expression?: string;
    age?: number;
    gender?: 'male' | 'female';
    landmarks?: Array<{ x: number; y: number }>;
    boundingBox?: { x: number; y: number; width: number; height: number };
  };
}

/**
 * Dataset metadata.
 */
export interface DatasetMetadata {
  name: string;
  version?: string;
  task: TrainingTask;
  classes?: string[];
  totalSamples: number;
  createdAt?: Date;
  source?: string;
}

/**
 * Options for loading from directory.
 */
export interface DirectoryOptions {
  /** Training task type */
  task: TrainingTask;
  /** Image extensions to include (default: ['jpg', 'jpeg', 'png', 'webp']) */
  extensions?: string[];
  /** Maximum samples per class (for balancing) */
  maxPerClass?: number;
  /** Recursive search (default: true) */
  recursive?: boolean;
}

/**
 * Options for loading from CSV.
 */
export interface CSVOptions {
  /** Column containing image path */
  imageColumn: string;
  /** Columns containing labels */
  labelColumns: string[];
  /** Base path for image files */
  basePath?: string;
  /** CSV delimiter (default: ',') */
  delimiter?: string;
  /** Skip header row (default: true) */
  hasHeader?: boolean;
}

/**
 * Options for creating TensorFlow dataset.
 */
export interface TensorDatasetOptions {
  /** Batch size */
  batchSize?: number;
  /** Shuffle buffer size */
  shuffleBuffer?: number;
  /** Prefetch buffer size */
  prefetchBuffer?: number;
  /** Image size [height, width] */
  imageSize?: [number, number];
  /** Normalize images to [0, 1] (default: true) */
  normalize?: boolean;
}

/**
 * Training Dataset.
 */
export class TrainingDataset {
  private _samples: TrainingSample[] = [];
  private _metadata: DatasetMetadata;

  constructor(metadata: Partial<DatasetMetadata> = {}) {
    this._metadata = {
      name: metadata.name || 'Untitled',
      task: metadata.task || 'recognition',
      totalSamples: 0,
      ...metadata
    };
  }

  /**
   * Add a sample to the dataset.
   */
  public addSample(sample: TrainingSample): void {
    this._samples.push(sample);
    this._metadata.totalSamples = this._samples.length;
  }

  /**
   * Add multiple samples.
   */
  public addSamples(samples: TrainingSample[]): void {
    this._samples.push(...samples);
    this._metadata.totalSamples = this._samples.length;
  }

  /**
   * Get all samples.
   */
  public getSamples(): TrainingSample[] {
    return this._samples;
  }

  /**
   * Get dataset metadata.
   */
  public get metadata(): DatasetMetadata {
    return { ...this._metadata };
  }

  /**
   * Get number of samples.
   */
  public get size(): number {
    return this._samples.length;
  }

  /**
   * Get unique classes (for classification tasks).
   */
  public getClasses(): string[] {
    const classes = new Set<string>();
    for (const sample of this._samples) {
      if (sample.labels.identity) classes.add(sample.labels.identity);
      if (sample.labels.expression) classes.add(sample.labels.expression);
    }
    return [...classes].sort();
  }

  /**
   * Split dataset into train/validation/test sets.
   */
  public split(trainRatio = 0.8, valRatio = 0.1): {
    train: TrainingDataset;
    validation: TrainingDataset;
    test: TrainingDataset;
  } {
    const shuffled = [...this._samples];
    tf.util.shuffle(shuffled);

    const trainEnd = Math.floor(shuffled.length * trainRatio);
    const valEnd = Math.floor(shuffled.length * (trainRatio + valRatio));

    const train = new TrainingDataset({ ...this._metadata, name: `${this._metadata.name}-train` });
    const validation = new TrainingDataset({ ...this._metadata, name: `${this._metadata.name}-val` });
    const test = new TrainingDataset({ ...this._metadata, name: `${this._metadata.name}-test` });

    train.addSamples(shuffled.slice(0, trainEnd));
    validation.addSamples(shuffled.slice(trainEnd, valEnd));
    test.addSamples(shuffled.slice(valEnd));

    return { train, validation, test };
  }

  /**
   * Shuffle samples in place.
   */
  public shuffle(): void {
    tf.util.shuffle(this._samples);
  }

  /**
   * Balance dataset by undersampling majority classes.
   */
  public balance(): void {
    const byClass = new Map<string, TrainingSample[]>();

    for (const sample of this._samples) {
      const key = sample.labels.identity || sample.labels.expression || 'unknown';
      if (!byClass.has(key)) byClass.set(key, []);
      byClass.get(key)!.push(sample);
    }

    const minCount = Math.min(...[...byClass.values()].map(s => s.length));

    this._samples = [];
    for (const samples of byClass.values()) {
      tf.util.shuffle(samples);
      this._samples.push(...samples.slice(0, minCount));
    }

    this._metadata.totalSamples = this._samples.length;
  }

  /**
   * Filter samples by predicate.
   */
  public filter(predicate: (sample: TrainingSample) => boolean): TrainingDataset {
    const filtered = new TrainingDataset(this._metadata);
    filtered.addSamples(this._samples.filter(predicate));
    return filtered;
  }

  /**
   * Create a TensorFlow.js dataset generator.
   */
  public toTensorDataset(options: TensorDatasetOptions = {}): tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }> {
    const {
      batchSize = 32,
      shuffleBuffer = 1000,
      prefetchBuffer = 2,
      imageSize = [224, 224],
      normalize = true
    } = options;

    const samples = this._samples;
    const task = this._metadata.task;

    const generator = function* () {
      for (const sample of samples) {
        yield sample;
      }
    };

    return tf.data.generator(generator)
      .shuffle(shuffleBuffer)
      .map((sample: TrainingSample) => {
        // This would need actual image loading - placeholder for now
        const xs = tf.zeros([imageSize[0], imageSize[1], 3]);
        const ys = tf.zeros([1]); // Placeholder

        return { xs, ys };
      })
      .batch(batchSize)
      .prefetch(prefetchBuffer);
  }

  /**
   * Export to JSON.
   */
  public toJSON(): { metadata: DatasetMetadata; samples: TrainingSample[] } {
    return {
      metadata: this._metadata,
      samples: this._samples.map(s => ({
        ...s,
        image: typeof s.image === 'string' ? s.image : '[Tensor]'
      }))
    };
  }

  /**
   * Save to JSON string.
   */
  public save(): string {
    return JSON.stringify(this.toJSON(), null, 2);
  }
}

/**
 * Dataset Loader for loading training data from various sources.
 */
export class DatasetLoader {
  /**
   * Load dataset from a directory with class folders.
   *
   * Expected structure:
   * ```
   * dataset/
   * ├── person1/
   * │   ├── img1.jpg
   * │   └── img2.jpg
   * └── person2/
   *     ├── img1.jpg
   *     └── img2.jpg
   * ```
   */
  public async fromDirectory(
    path: string,
    options: DirectoryOptions
  ): Promise<TrainingDataset> {
    const {
      task,
      extensions = ['jpg', 'jpeg', 'png', 'webp'],
      maxPerClass
    } = options;

    const dataset = new TrainingDataset({
      name: path.split('/').pop() || 'Dataset',
      task,
      source: path
    });

    // In browser, this would need a file input
    // In Node.js, use fs to read directory
    if (typeof process !== 'undefined' && process.versions?.node) {
      const fs = await import('fs').catch(() => null);
      const pathModule = await import('path').catch(() => null);

      if (fs && pathModule) {
        const classes = fs.readdirSync(path).filter((f: string) =>
          fs.statSync(pathModule.join(path, f)).isDirectory()
        );

        for (const className of classes) {
          const classPath = pathModule.join(path, className);
          let files = fs.readdirSync(classPath).filter((f: string) =>
            extensions.some(ext => f.toLowerCase().endsWith(`.${ext}`))
          );

          if (maxPerClass) {
            files = files.slice(0, maxPerClass);
          }

          for (const file of files) {
            dataset.addSample({
              image: pathModule.join(classPath, file),
              labels: { identity: className }
            });
          }
        }
      }
    }

    return dataset;
  }

  /**
   * Load dataset from CSV file.
   */
  public async fromCSV(
    path: string,
    options: CSVOptions
  ): Promise<TrainingDataset> {
    const {
      imageColumn,
      labelColumns,
      basePath = '',
      delimiter = ',',
      hasHeader = true
    } = options;

    const dataset = new TrainingDataset({
      name: path.split('/').pop()?.replace('.csv', '') || 'Dataset',
      task: 'recognition',
      source: path
    });

    // Read and parse CSV
    let content: string;

    if (typeof process !== 'undefined' && process.versions?.node) {
      const fs = await import('fs').catch(() => null);
      if (fs) {
        content = fs.readFileSync(path, 'utf-8');
      } else {
        throw new Error('Cannot read file in this environment');
      }
    } else if (typeof fetch !== 'undefined') {
      const response = await fetch(path);
      content = await response.text();
    } else {
      throw new Error('Cannot read file in this environment');
    }

    const lines = content.split('\n').filter(l => l.trim());
    const headers = hasHeader
      ? lines[0].split(delimiter).map(h => h.trim())
      : [];
    const dataLines = hasHeader ? lines.slice(1) : lines;

    const imageIdx = hasHeader ? headers.indexOf(imageColumn) : 0;
    const labelIndices = hasHeader
      ? labelColumns.map(c => headers.indexOf(c))
      : labelColumns.map((_, i) => i + 1);

    for (const line of dataLines) {
      const values = line.split(delimiter).map(v => v.trim());

      const imagePath = basePath
        ? `${basePath}/${values[imageIdx]}`
        : values[imageIdx];

      const labels: TrainingSample['labels'] = {};

      for (let i = 0; i < labelColumns.length; i++) {
        const column = labelColumns[i];
        const value = values[labelIndices[i]];

        if (column === 'identity' || column === 'expression') {
          labels[column] = value;
        } else if (column === 'age') {
          labels.age = parseFloat(value);
        } else if (column === 'gender') {
          labels.gender = value as 'male' | 'female';
        }
      }

      dataset.addSample({ image: imagePath, labels });
    }

    return dataset;
  }

  /**
   * Load dataset from JSON manifest.
   */
  public async fromManifest(path: string): Promise<TrainingDataset> {
    let content: string;

    if (typeof process !== 'undefined' && process.versions?.node) {
      const fs = await import('fs').catch(() => null);
      if (fs) {
        content = fs.readFileSync(path, 'utf-8');
      } else {
        throw new Error('Cannot read file in this environment');
      }
    } else if (typeof fetch !== 'undefined') {
      const response = await fetch(path);
      content = await response.text();
    } else {
      throw new Error('Cannot read file in this environment');
    }

    const data = JSON.parse(content);
    const dataset = new TrainingDataset(data.metadata);
    dataset.addSamples(data.samples);

    return dataset;
  }

  /**
   * Load dataset from URLs.
   */
  public async fromURLs(
    urls: string[],
    labels: TrainingSample['labels'][],
    options: { task?: TrainingTask } = {}
  ): Promise<TrainingDataset> {
    const dataset = new TrainingDataset({
      name: 'URL Dataset',
      task: options.task || 'recognition',
      source: 'urls'
    });

    for (let i = 0; i < urls.length; i++) {
      dataset.addSample({
        image: urls[i],
        labels: labels[i] || {}
      });
    }

    return dataset;
  }
}

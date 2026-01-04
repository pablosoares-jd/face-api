/**
 * Model Trainer for fine-tuning face-api models.
 *
 * Supports:
 * - Transfer learning with frozen feature extractors
 * - Full model fine-tuning
 * - Custom loss functions
 * - Training callbacks
 *
 * @example
 * ```typescript
 * import { ModelTrainer } from '@vladmandic/face-api/training';
 * import * as faceapi from '@vladmandic/face-api';
 *
 * // Load model
 * await faceapi.nets.ageGenderNet.loadFromUri('/models');
 *
 * // Create trainer
 * const trainer = new ModelTrainer(faceapi.nets.ageGenderNet);
 * trainer.freezeFeatureExtractor();
 *
 * // Train
 * await trainer.fit(images, labels, {
 *   epochs: 10,
 *   batchSize: 32,
 *   learningRate: 0.0001
 * });
 *
 * // Save
 * await trainer.saveWeights('custom-age-gender');
 * ```
 */

import * as tf from '@tensorflow/tfjs';
import { NeuralNetwork } from '../NeuralNetwork';
import {
  EarlyStopping,
  EarlyStoppingConfig,
  LRScheduler,
  LRSchedulerConfig,
  Checkpointer,
  CheckpointConfig,
  GradientClipConfig,
  createAdamW,
  TrainingProgress
} from './TrainingUtils';

/**
 * Training configuration.
 */
export interface TrainingConfig {
  /** Number of training epochs (default: 10) */
  epochs?: number;
  /** Batch size (default: 32) */
  batchSize?: number;
  /** Learning rate (default: 0.0001) */
  learningRate?: number;
  /** Validation split ratio (default: 0.2) */
  validationSplit?: number;
  /** Shuffle data each epoch (default: true) */
  shuffle?: boolean;
  /** Optimizer type (default: 'adam') */
  optimizer?: 'adam' | 'adamw' | 'sgd' | 'rmsprop' | 'adagrad';
  /** Weight decay for AdamW (default: 0.01) */
  weightDecay?: number;
  /** Loss function (default: auto-detected) */
  loss?: 'mse' | 'categoricalCrossentropy' | 'binaryCrossentropy' | 'custom';
  /** Custom loss function */
  customLoss?: (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.Scalar;
  /** Training callbacks */
  callbacks?: TrainingCallbacks;
  /** Early stopping configuration */
  earlyStopping?: EarlyStoppingConfig;
  /** Learning rate scheduler configuration */
  lrScheduler?: LRSchedulerConfig;
  /** Checkpointing configuration */
  checkpointing?: CheckpointConfig;
  /** Gradient clipping configuration */
  gradientClip?: GradientClipConfig;
}

/**
 * Training callbacks for monitoring progress.
 */
export interface TrainingCallbacks {
  onTrainBegin?: () => void | Promise<void>;
  onTrainEnd?: (logs: TrainingLogs) => void | Promise<void>;
  onEpochBegin?: (epoch: number) => void | Promise<void>;
  onEpochEnd?: (epoch: number, logs: EpochLogs) => void | Promise<void>;
  onBatchBegin?: (batch: number) => void | Promise<void>;
  onBatchEnd?: (batch: number, logs: BatchLogs) => void | Promise<void>;
}

/**
 * Training logs.
 */
export interface TrainingLogs {
  epochs: number;
  totalBatches: number;
  finalLoss: number;
  finalValLoss?: number;
  history: EpochLogs[];
  /** Whether training stopped early */
  stoppedEarly?: boolean;
}

/**
 * Epoch logs.
 */
export interface EpochLogs {
  epoch: number;
  loss: number;
  valLoss?: number;
  accuracy?: number;
  valAccuracy?: number;
  learningRate: number;
  duration: number;
}

/**
 * Batch logs.
 */
export interface BatchLogs {
  batch: number;
  loss: number;
  size: number;
}

/**
 * Model Trainer for fine-tuning neural networks.
 */
export class ModelTrainer<TNetParams> {
  private _model: NeuralNetwork<TNetParams>;
  private _optimizer: tf.Optimizer | null = null;
  private _isCompiled = false;
  private _frozenPaths: Set<string> = new Set();

  constructor(model: NeuralNetwork<TNetParams>) {
    if (!model.isLoaded) {
      throw new Error('Model must be loaded before training');
    }
    this._model = model;
  }

  /**
   * Freeze all parameters (make non-trainable).
   */
  public freeze(): void {
    this._model.freeze();
    for (const { path } of this._model.getParamList()) {
      this._frozenPaths.add(path);
    }
  }

  /**
   * Unfreeze all parameters (make trainable).
   */
  public unfreeze(): void {
    this._model.variable();
    this._frozenPaths.clear();
  }

  /**
   * Freeze feature extractor layers only.
   * Keeps classification head trainable for transfer learning.
   */
  public async freezeFeatureExtractor(): Promise<void> {
    const params = this._model.getParamList();

    // Common patterns for feature extractor layers
    const featurePatterns = [
      /conv/i,
      /bn/i,
      /batch_norm/i,
      /dense_block/i,
      /residual/i,
      /backbone/i,
      /encoder/i
    ];

    for (const { path, tensor } of params) {
      const isFeatureExtractor = featurePatterns.some(p => p.test(path));

      if (isFeatureExtractor && tensor instanceof tf.Variable) {
        // Convert to frozen tensor (async to avoid blocking)
        const data = await tensor.data();
        const frozen = tf.tensor(data, tensor.shape);
        tensor.dispose();
        this._model.reassignParamFromPath(path, frozen);
        this._frozenPaths.add(path);
      }
    }
  }

  /**
   * Freeze specific layers by path pattern.
   */
  public async freezeLayers(pattern: RegExp): Promise<void> {
    const params = this._model.getParamList();

    for (const { path, tensor } of params) {
      if (pattern.test(path) && tensor instanceof tf.Variable) {
        const data = await tensor.data();
        const frozen = tf.tensor(data, tensor.shape);
        tensor.dispose();
        this._model.reassignParamFromPath(path, frozen);
        this._frozenPaths.add(path);
      }
    }
  }

  /**
   * Get trainable parameters.
   */
  public getTrainableParams(): Array<{ path: string; tensor: tf.Tensor }> {
    return this._model.getTrainableParams();
  }

  /**
   * Get frozen parameters.
   */
  public getFrozenParams(): Array<{ path: string; tensor: tf.Tensor }> {
    return this._model.getFrozenParams();
  }

  /**
   * Compile the model for training.
   */
  public compile(config: Partial<TrainingConfig> = {}): void {
    const { optimizer = 'adam', learningRate = 0.0001, weightDecay = 0.01 } = config;

    // Make parameters trainable (if not frozen)
    const params = this._model.getParamList();
    for (const { path, tensor } of params) {
      if (!this._frozenPaths.has(path) && !(tensor instanceof tf.Variable)) {
        this._model.reassignParamFromPath(path, tensor.variable());
      }
    }

    // Create optimizer
    switch (optimizer) {
      case 'sgd':
        this._optimizer = tf.train.sgd(learningRate);
        break;
      case 'rmsprop':
        this._optimizer = tf.train.rmsprop(learningRate);
        break;
      case 'adagrad':
        this._optimizer = tf.train.adagrad(learningRate);
        break;
      case 'adamw':
        this._optimizer = createAdamW({ learningRate, weightDecay });
        break;
      case 'adam':
      default:
        this._optimizer = tf.train.adam(learningRate);
        break;
    }

    this._isCompiled = true;
  }

  /**
   * Train the model.
   *
   * @param inputs - Training inputs (images or tensors)
   * @param labels - Training labels
   * @param config - Training configuration
   */
  public async fit(
    inputs: tf.Tensor | tf.Tensor[],
    labels: tf.Tensor | tf.Tensor[],
    config: TrainingConfig = {}
  ): Promise<TrainingLogs> {
    const {
      epochs = 10,
      batchSize = 32,
      learningRate = 0.0001,
      validationSplit = 0.2,
      shuffle = true,
      loss = 'mse',
      customLoss,
      callbacks = {},
      earlyStopping: earlyStoppingConfig,
      lrScheduler: lrSchedulerConfig,
      checkpointing: checkpointingConfig
    } = config;

    if (!this._isCompiled) {
      this.compile({ ...config, learningRate });
    }

    const trainableParams = this.getTrainableParams();
    if (trainableParams.length === 0) {
      throw new Error('No trainable parameters. Call unfreeze() or check model state.');
    }

    // Initialize training utilities
    const earlyStopping = earlyStoppingConfig
      ? new EarlyStopping(earlyStoppingConfig)
      : null;

    const lrScheduler = lrSchedulerConfig
      ? new LRScheduler(lrSchedulerConfig)
      : null;

    const checkpointer = checkpointingConfig
      ? new Checkpointer(checkpointingConfig)
      : null;

    const progress = new TrainingProgress();

    // Get loss function
    const lossFunction = customLoss || this._getLossFunction(loss);

    // Prepare data - clone to avoid disposing user's tensors
    const inputTensor = Array.isArray(inputs) ? tf.concat(inputs) : inputs.clone();
    const labelTensor = Array.isArray(labels) ? tf.concat(labels) : labels.clone();

    const numSamples = inputTensor.shape[0];
    const numValidation = Math.floor(numSamples * validationSplit);
    const numTrain = numSamples - numValidation;

    // Split train/validation
    const indices = tf.util.createShuffledIndices(numSamples);
    const trainIndices = indices.slice(0, numTrain);
    const valIndices = indices.slice(numTrain);

    await callbacks.onTrainBegin?.();
    progress.start();

    const history: EpochLogs[] = [];
    let totalBatches = 0;
    let stoppedEarly = false;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const epochStart = Date.now();
      await callbacks.onEpochBegin?.(epoch);

      // Get current learning rate from scheduler
      const currentLR = lrScheduler
        ? lrScheduler.getLR(epoch, history[history.length - 1]?.valLoss)
        : learningRate;

      // Shuffle training indices
      if (shuffle) {
        tf.util.shuffle(trainIndices);
      }

      let epochLoss = 0;
      let batchCount = 0;

      // Training loop
      for (let i = 0; i < numTrain; i += batchSize) {
        const batchEnd = Math.min(i + batchSize, numTrain);
        const batchIndices = trainIndices.slice(i, batchEnd);

        await callbacks.onBatchBegin?.(batchCount);

        const batchLoss = this._optimizer!.minimize(() => {
          const batchInputs = tf.gather(inputTensor, batchIndices);
          const batchLabels = tf.gather(labelTensor, batchIndices);

          // Forward pass (model-specific, needs to be implemented per model)
          const predictions = this._forwardPass(batchInputs);

          // Compute loss
          const loss = lossFunction(batchLabels, predictions);

          // Cleanup
          batchInputs.dispose();
          batchLabels.dispose();
          predictions.dispose();

          return loss;
        }, true) as tf.Scalar;

        const lossData = await batchLoss.data();
        const lossValue = lossData[0];
        epochLoss += lossValue;
        batchLoss.dispose();

        await callbacks.onBatchEnd?.(batchCount, {
          batch: batchCount,
          loss: lossValue,
          size: batchEnd - i
        });

        batchCount++;
        totalBatches++;
      }

      epochLoss /= batchCount;

      // Validation
      let valLoss: number | undefined;
      if (numValidation > 0) {
        const valInputs = tf.gather(inputTensor, valIndices);
        const valLabels = tf.gather(labelTensor, valIndices);

        const valPredictions = this._forwardPass(valInputs);
        const valLossTensor = lossFunction(valLabels, valPredictions);
        const valLossData = await valLossTensor.data();
        valLoss = valLossData[0];

        valInputs.dispose();
        valLabels.dispose();
        valPredictions.dispose();
        valLossTensor.dispose();
      }

      const epochDuration = Date.now() - epochStart;

      const epochLogs: EpochLogs = {
        epoch,
        loss: epochLoss,
        valLoss,
        learningRate: currentLR,
        duration: epochDuration
      };

      history.push(epochLogs);
      progress.recordEpoch(epochLogs);

      // Checkpointing
      if (checkpointer) {
        const weights = await this._model.serializeParams();
        checkpointer.maybeSave(epoch, epochLogs, weights);
      }

      await callbacks.onEpochEnd?.(epoch, epochLogs);

      // Calculate ETA
      const eta = progress.getETA(epoch, epochs);
      const etaStr = eta > 0 ? ` - ETA: ${TrainingProgress.formatTime(eta)}` : '';

      console.log(
        `Epoch ${epoch + 1}/${epochs} - ` +
        `loss: ${epochLoss.toFixed(4)}` +
        (valLoss !== undefined ? ` - val_loss: ${valLoss.toFixed(4)}` : '') +
        (lrScheduler ? ` - lr: ${currentLR.toExponential(2)}` : '') +
        ` - ${epochDuration}ms${etaStr}`
      );

      // Early stopping check
      if (earlyStopping?.check(epoch, epochLogs)) {
        stoppedEarly = true;
        console.log(`\nTraining stopped early at epoch ${epoch + 1}`);
        break;
      }
    }

    const actualEpochs = stoppedEarly ? history.length : epochs;

    const finalLogs: TrainingLogs = {
      epochs: actualEpochs,
      totalBatches,
      finalLoss: history[history.length - 1].loss,
      finalValLoss: history[history.length - 1].valLoss,
      history,
      stoppedEarly
    };

    await callbacks.onTrainEnd?.(finalLogs);

    // Print training summary
    const summary = progress.getSummary();
    console.log('\n=== Training Summary ===');
    console.log(`Total time: ${TrainingProgress.formatTime(summary.totalTime)}`);
    console.log(`Epochs completed: ${actualEpochs}${stoppedEarly ? ' (early stopped)' : ''}`);
    console.log(`Best loss: ${summary.bestLoss.toFixed(6)}`);
    if (summary.bestValLoss !== undefined) {
      console.log(`Best val_loss: ${summary.bestValLoss.toFixed(6)}`);
    }

    // Cleanup
    inputTensor.dispose();
    labelTensor.dispose();

    return finalLogs;
  }

  /**
   * Forward pass through the model.
   * Override this for specific model architectures.
   */
  protected _forwardPass(inputs: tf.Tensor): tf.Tensor {
    // This is a placeholder - actual implementation depends on model type
    // For now, return inputs (should be overridden)
    return inputs.clone();
  }

  /**
   * Get loss function by name.
   */
  private _getLossFunction(loss: string): (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.Scalar {
    switch (loss) {
      case 'mse':
        return (yTrue, yPred) => tf.losses.meanSquaredError(yTrue, yPred);
      case 'categoricalCrossentropy':
        return (yTrue, yPred) => tf.losses.softmaxCrossEntropy(yTrue, yPred);
      case 'binaryCrossentropy':
        return (yTrue, yPred) => tf.losses.sigmoidCrossEntropy(yTrue, yPred);
      default:
        return (yTrue, yPred) => tf.losses.meanSquaredError(yTrue, yPred);
    }
  }

  /**
   * Save model weights with training metadata.
   */
  public async saveWeights(
    path: string,
    trainingLogs?: TrainingLogs
  ): Promise<void> {
    const serialized = await this._model.serializeWithMetadata({
      epochs: trainingLogs?.epochs,
      finalLoss: trainingLogs?.finalLoss
    });

    const json = JSON.stringify(serialized);

    // In browser, save to localStorage or download
    if (typeof window !== 'undefined') {
      localStorage.setItem(`faceapi-weights-${path}`, json);
    }

    // In Node.js, save to file
    if (typeof process !== 'undefined' && process.versions?.node) {
      const fs = await import('fs').catch(() => null);
      if (fs) {
        fs.writeFileSync(`${path}.json`, json);
      }
    }
  }

  /**
   * Load model weights from versioned format.
   */
  public async loadWeights(path: string): Promise<void> {
    let json: string;

    // In browser, load from localStorage
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(`faceapi-weights-${path}`);
      if (stored) {
        json = stored;
      } else {
        throw new Error(`Weights not found: ${path}`);
      }
    } else if (typeof process !== 'undefined' && process.versions?.node) {
      // In Node.js, load from file
      const fs = await import('fs').catch(() => null);
      if (fs && fs.existsSync(`${path}.json`)) {
        json = fs.readFileSync(`${path}.json`, 'utf-8');
      } else {
        throw new Error(`Weights file not found: ${path}.json`);
      }
    } else {
      throw new Error('Cannot load weights in this environment');
    }

    const data = JSON.parse(json);
    this._model.loadFromSerialized(data);
  }

  /**
   * Get current model weights as Float32Array.
   */
  public async getWeights(): Promise<Float32Array> {
    return this._model.serializeParams();
  }

  /**
   * Dispose optimizer and cleanup.
   */
  public dispose(): void {
    if (this._optimizer) {
      this._optimizer.dispose();
      this._optimizer = null;
    }
    this._isCompiled = false;
  }
}

/**
 * Triplet loss for face embedding training.
 *
 * @param anchor - Anchor embeddings
 * @param positive - Positive embeddings (same identity)
 * @param negative - Negative embeddings (different identity)
 * @param margin - Margin for triplet loss (default: 0.2)
 */
export function tripletLoss(
  anchor: tf.Tensor,
  positive: tf.Tensor,
  negative: tf.Tensor,
  margin = 0.2
): tf.Scalar {
  return tf.tidy(() => {
    const positiveDist = tf.sum(tf.square(tf.sub(anchor, positive)), -1);
    const negativeDist = tf.sum(tf.square(tf.sub(anchor, negative)), -1);
    const loss = tf.maximum(0, tf.add(tf.sub(positiveDist, negativeDist), margin));
    return tf.mean(loss) as tf.Scalar;
  });
}

/**
 * Contrastive loss for siamese networks.
 *
 * @param embeddings1 - First set of embeddings
 * @param embeddings2 - Second set of embeddings
 * @param labels - 1 for same identity, 0 for different
 * @param margin - Margin for different pairs (default: 1.0)
 */
export function contrastiveLoss(
  embeddings1: tf.Tensor,
  embeddings2: tf.Tensor,
  labels: tf.Tensor,
  margin = 1.0
): tf.Scalar {
  return tf.tidy(() => {
    const distances = tf.sqrt(tf.sum(tf.square(tf.sub(embeddings1, embeddings2)), -1));

    // Loss for same pairs: distance^2
    const sameLoss = tf.mul(labels, tf.square(distances));

    // Loss for different pairs: max(0, margin - distance)^2
    const diffLoss = tf.mul(
      tf.sub(1, labels),
      tf.square(tf.maximum(0, tf.sub(margin, distances)))
    );

    return tf.mean(tf.add(sameLoss, diffLoss)) as tf.Scalar;
  });
}

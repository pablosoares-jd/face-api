/**
 * Training Utilities
 *
 * Advanced training features:
 * - Early Stopping
 * - Learning Rate Schedulers
 * - Gradient Clipping
 * - Checkpointing
 * - AdamW Optimizer
 */

import * as tf from '@tensorflow/tfjs';
import type { EpochLogs, TrainingCallbacks } from './ModelTrainer';

/**
 * Early stopping configuration.
 */
export interface EarlyStoppingConfig {
  /** Metric to monitor (default: 'valLoss') */
  monitor?: 'loss' | 'valLoss';
  /** Minimum improvement to reset patience (default: 0.001) */
  minDelta?: number;
  /** Number of epochs without improvement before stopping (default: 5) */
  patience?: number;
  /** Restore best weights when stopping (default: true) */
  restoreBestWeights?: boolean;
  /** Verbose logging (default: true) */
  verbose?: boolean;
}

/**
 * Early Stopping callback to prevent overfitting.
 */
export class EarlyStopping {
  private _config: Required<EarlyStoppingConfig>;
  private _bestValue = Infinity;
  private _bestEpoch = 0;
  private _waitCount = 0;
  private _bestWeights: Float32Array | null = null;
  private _stopped = false;

  constructor(config: EarlyStoppingConfig = {}) {
    this._config = {
      monitor: config.monitor ?? 'valLoss',
      minDelta: config.minDelta ?? 0.001,
      patience: config.patience ?? 5,
      restoreBestWeights: config.restoreBestWeights ?? true,
      verbose: config.verbose ?? true,
    };
  }

  /**
   * Check if training should stop.
   */
  public check(epoch: number, logs: EpochLogs): boolean {
    const currentValue = this._config.monitor === 'valLoss'
      ? logs.valLoss ?? logs.loss
      : logs.loss;

    if (currentValue < this._bestValue - this._config.minDelta) {
      this._bestValue = currentValue;
      this._bestEpoch = epoch;
      this._waitCount = 0;

      if (this._config.verbose) {
        console.info(`EarlyStopping: ${this._config.monitor} improved to ${currentValue.toFixed(6)}`);
      }
      return false;
    }

    this._waitCount++;

    if (this._waitCount >= this._config.patience) {
      this._stopped = true;
      if (this._config.verbose) {
        console.info(
          `EarlyStopping: Stopped at epoch ${epoch + 1}. `
          + `Best ${this._config.monitor}: ${this._bestValue.toFixed(6)} at epoch ${this._bestEpoch + 1}`,
        );
      }
      return true;
    }

    if (this._config.verbose) {
      console.info(
        `EarlyStopping: No improvement for ${this._waitCount}/${this._config.patience} epochs`,
      );
    }

    return false;
  }

  /**
   * Store current weights as best.
   */
  public saveBestWeights(weights: Float32Array): void {
    this._bestWeights = new Float32Array(weights);
  }

  /**
   * Get best weights if available.
   */
  public getBestWeights(): Float32Array | null {
    return this._bestWeights;
  }

  /**
   * Check if training was stopped early.
   */
  public get stopped(): boolean {
    return this._stopped;
  }

  /**
   * Get best epoch number.
   */
  public get bestEpoch(): number {
    return this._bestEpoch;
  }

  /**
   * Reset state for new training run.
   */
  public reset(): void {
    this._bestValue = Infinity;
    this._bestEpoch = 0;
    this._waitCount = 0;
    this._bestWeights = null;
    this._stopped = false;
  }
}

/**
 * Learning rate schedule type.
 */
export type ScheduleType =
  | 'constant'
  | 'step'
  | 'exponential'
  | 'cosine'
  | 'warmup'
  | 'reduceonplateau';

/**
 * Learning rate scheduler configuration.
 */
export interface LRSchedulerConfig {
  /** Schedule type */
  type: ScheduleType;
  /** Initial learning rate */
  initialLR: number;
  /** For step decay: epochs between decay */
  stepSize?: number;
  /** Decay factor (default: 0.1) */
  decayRate?: number;
  /** For cosine: total epochs */
  totalEpochs?: number;
  /** For warmup: warmup epochs */
  warmupEpochs?: number;
  /** Minimum learning rate (default: 1e-7) */
  minLR?: number;
  /** For reduce on plateau: patience */
  patience?: number;
}

/**
 * Learning Rate Scheduler.
 */
export class LRScheduler {
  private _config: LRSchedulerConfig;
  private _currentLR: number;
  private _bestLoss = Infinity;
  private _waitCount = 0;

  constructor(config: LRSchedulerConfig) {
    this._config = {
      decayRate: 0.1,
      minLR: 1e-7,
      patience: 3,
      ...config,
    };
    this._currentLR = config.initialLR;
  }

  /**
   * Get learning rate for given epoch.
   */
  public getLR(epoch: number, currentLoss?: number): number {
    const { type, initialLR, stepSize, decayRate, totalEpochs, warmupEpochs, minLR } = this._config;

    let lr = initialLR;

    switch (type) {
      case 'constant':
        lr = initialLR;
        break;

      case 'step':
        if (stepSize) {
          const numDecays = Math.floor(epoch / stepSize);
          lr = initialLR * (decayRate!) ** numDecays;
        }
        break;

      case 'exponential':
        lr = initialLR * (decayRate!) ** epoch;
        break;

      case 'cosine':
        if (totalEpochs) {
          lr = minLR! + 0.5 * (initialLR - minLR!)
            * (1 + Math.cos(Math.PI * epoch / totalEpochs));
        }
        break;

      case 'warmup':
        if (warmupEpochs && epoch < warmupEpochs) {
          lr = initialLR * (epoch + 1) / warmupEpochs;
        } else if (totalEpochs) {
          const adjustedEpoch = epoch - (warmupEpochs || 0);
          const adjustedTotal = totalEpochs - (warmupEpochs || 0);
          lr = minLR! + 0.5 * (initialLR - minLR!)
            * (1 + Math.cos(Math.PI * adjustedEpoch / adjustedTotal));
        }
        break;

      case 'reduceonplateau':
        if (currentLoss !== undefined) {
          if (currentLoss < this._bestLoss) {
            this._bestLoss = currentLoss;
            this._waitCount = 0;
          } else {
            this._waitCount++;
            if (this._waitCount >= (this._config.patience || 3)) {
              this._currentLR = Math.max(this._currentLR * decayRate!, minLR!);
              this._waitCount = 0;
              console.info(`LRScheduler: Reduced LR to ${this._currentLR.toExponential(2)}`);
            }
          }
          lr = this._currentLR;
        }
        break;
    }

    this._currentLR = Math.max(lr, minLR!);
    return this._currentLR;
  }

  /**
   * Get current learning rate.
   */
  public get currentLR(): number {
    return this._currentLR;
  }

  /**
   * Reset scheduler state.
   */
  public reset(): void {
    this._currentLR = this._config.initialLR;
    this._bestLoss = Infinity;
    this._waitCount = 0;
  }
}

/**
 * Gradient clipping configuration.
 */
export interface GradientClipConfig {
  /** Clip by value (clips each gradient element) */
  clipValue?: number;
  /** Clip by global norm (scales gradients if total norm exceeds) */
  clipNorm?: number;
}

/**
 * Clip gradients to prevent exploding gradients.
 */
export function clipGradients(
  gradients: tf.NamedTensorMap,
  config: GradientClipConfig,
): tf.NamedTensorMap {
  return tf.tidy(() => {
    const clipped: tf.NamedTensorMap = {};

    if (config.clipValue !== undefined) {
      // Clip by value
      for (const [name, grad] of Object.entries(gradients)) {
        clipped[name] = tf.clipByValue(grad, -config.clipValue, config.clipValue);
      }
    } else if (config.clipNorm !== undefined) {
      // Clip by global norm
      const grads = Object.values(gradients);
      const squaredNorms = grads.map((g) => tf.sum(tf.square(g)));
      const globalNorm = tf.sqrt(tf.addN(squaredNorms));
      const globalNormValue = globalNorm.dataSync()[0] ?? 0;

      const scale = globalNormValue > config.clipNorm
        ? config.clipNorm / globalNormValue
        : 1.0;

      for (const [name, grad] of Object.entries(gradients)) {
        clipped[name] = tf.mul(grad, scale);
      }
    } else {
      // No clipping
      for (const [name, grad] of Object.entries(gradients)) {
        clipped[name] = grad.clone();
      }
    }

    return clipped;
  });
}

/**
 * Checkpointing configuration.
 */
export interface CheckpointConfig {
  /** Save every N epochs (default: 1) */
  saveFrequency?: number;
  /** Maximum checkpoints to keep (default: 3) */
  maxToKeep?: number;
  /** Save only when metric improves (default: true) */
  saveBestOnly?: boolean;
  /** Metric to monitor (default: 'valLoss') */
  monitor?: 'loss' | 'valLoss';
  /** Checkpoint name prefix */
  prefix?: string;
}

/**
 * Model checkpointing during training.
 */
export class Checkpointer {
  private _config: Required<CheckpointConfig>;
  private _bestValue = Infinity;
  private _checkpoints: Array<{ epoch: number; value: number; weights: Float32Array }> = [];

  constructor(config: CheckpointConfig = {}) {
    this._config = {
      saveFrequency: config.saveFrequency ?? 1,
      maxToKeep: config.maxToKeep ?? 3,
      saveBestOnly: config.saveBestOnly ?? true,
      monitor: config.monitor ?? 'valLoss',
      prefix: config.prefix ?? 'checkpoint',
    };
  }

  /**
   * Check if should save checkpoint and save if needed.
   */
  public maybeSave(epoch: number, logs: EpochLogs, weights: Float32Array): boolean {
    if (epoch % this._config.saveFrequency !== 0) {
      return false;
    }

    const currentValue = this._config.monitor === 'valLoss'
      ? logs.valLoss ?? logs.loss
      : logs.loss;

    if (this._config.saveBestOnly) {
      if (currentValue >= this._bestValue) {
        return false;
      }
      this._bestValue = currentValue;
    }

    // Save checkpoint
    this._checkpoints.push({
      epoch,
      value: currentValue,
      weights: new Float32Array(weights),
    });

    // Remove old checkpoints if exceeding max
    while (this._checkpoints.length > this._config.maxToKeep) {
      this._checkpoints.shift();
    }

    console.info(
      `Checkpoint: Saved epoch ${epoch + 1} (${this._config.monitor}: ${currentValue.toFixed(6)})`,
    );

    return true;
  }

  /**
   * Get best checkpoint.
   */
  public getBest(): { epoch: number; value: number; weights: Float32Array } | null {
    if (this._checkpoints.length === 0) return null;

    return this._checkpoints.reduce((best, curr) => (curr.value < best.value ? curr : best));
  }

  /**
   * Get latest checkpoint.
   */
  public getLatest(): { epoch: number; value: number; weights: Float32Array } | null {
    if (this._checkpoints.length === 0) return null;
    return this._checkpoints[this._checkpoints.length - 1] ?? null;
  }

  /**
   * Get all checkpoints.
   */
  public getAll(): Array<{ epoch: number; value: number }> {
    return this._checkpoints.map((c) => ({ epoch: c.epoch, value: c.value }));
  }

  /**
   * Clear all checkpoints.
   */
  public clear(): void {
    this._checkpoints = [];
    this._bestValue = Infinity;
  }
}

/**
 * AdamW optimizer configuration.
 */
export interface AdamWConfig {
  /** Learning rate (default: 0.001) */
  learningRate?: number;
  /** Beta1 for first moment (default: 0.9) */
  beta1?: number;
  /** Beta2 for second moment (default: 0.999) */
  beta2?: number;
  /** Epsilon for numerical stability (default: 1e-7) */
  epsilon?: number;
  /** Weight decay coefficient (default: 0.01) */
  weightDecay?: number;
}

/**
 * Create AdamW optimizer (Adam with decoupled weight decay).
 *
 * Weight decay is applied separately from gradient updates,
 * which often works better than L2 regularization.
 */
export function createAdamW(config: AdamWConfig = {}): tf.Optimizer {
  const {
    learningRate = 0.001,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-7,
    weightDecay = 0.01,
  } = config;

  // Create base Adam optimizer
  const adam = tf.train.adam(learningRate, beta1, beta2, epsilon);

  // Wrap with weight decay
  const originalMinimize = adam.minimize.bind(adam);

  adam.minimize = (f: () => tf.Scalar, returnCost?: boolean, varList?: tf.Variable[]) => {
    const result = originalMinimize(f, returnCost, varList);

    // Apply weight decay to all variables
    const variables = varList || (tf.engine().registeredVariables as unknown as tf.Variable[]);

    tf.tidy(() => {
      for (const variable of Object.values(variables)) {
        if (variable instanceof tf.Variable) {
          const decayed = tf.sub(variable, tf.mul(variable, learningRate * weightDecay));
          variable.assign(decayed);
        }
      }
    });

    return result;
  };

  return adam;
}

/**
 * Training progress tracker.
 */
export class TrainingProgress {
  private _startTime: number = 0;
  private _epochTimes: number[] = [];
  private _losses: number[] = [];
  private _valLosses: number[] = [];

  /**
   * Start tracking.
   */
  public start(): void {
    this._startTime = Date.now();
    this._epochTimes = [];
    this._losses = [];
    this._valLosses = [];
  }

  /**
   * Record epoch metrics.
   */
  public recordEpoch(logs: EpochLogs): void {
    this._epochTimes.push(logs.duration);
    this._losses.push(logs.loss);
    if (logs.valLoss !== undefined) {
      this._valLosses.push(logs.valLoss);
    }
  }

  /**
   * Get estimated time remaining.
   */
  public getETA(currentEpoch: number, totalEpochs: number): number {
    if (this._epochTimes.length === 0) return 0;

    const avgEpochTime = this._epochTimes.reduce((a, b) => a + b, 0) / this._epochTimes.length;
    return avgEpochTime * (totalEpochs - currentEpoch - 1);
  }

  /**
   * Get total elapsed time.
   */
  public getElapsed(): number {
    return Date.now() - this._startTime;
  }

  /**
   * Get training summary.
   */
  public getSummary(): {
    totalTime: number;
    avgEpochTime: number;
    bestLoss: number;
    bestValLoss: number | undefined;
    finalLoss: number;
    finalValLoss: number | undefined;
    } {
    return {
      totalTime: this.getElapsed(),
      avgEpochTime: this._epochTimes.length > 0
        ? this._epochTimes.reduce((a, b) => a + b, 0) / this._epochTimes.length
        : 0,
      bestLoss: Math.min(...this._losses),
      bestValLoss: this._valLosses.length > 0 ? Math.min(...this._valLosses) : undefined,
      finalLoss: this._losses[this._losses.length - 1] ?? 0,
      finalValLoss: this._valLosses[this._valLosses.length - 1],
    };
  }

  /**
   * Format time in human readable format.
   */
  public static formatTime(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
    return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
  }
}

/**
 * Create training callbacks from utilities.
 */
export function createCallbacks(options: {
  earlyStopping?: EarlyStoppingConfig;
  lrScheduler?: LRSchedulerConfig;
  checkpointing?: CheckpointConfig;
  verbose?: boolean;
}): {
  callbacks: TrainingCallbacks;
  earlyStopping?: EarlyStopping;
  lrScheduler?: LRScheduler;
  checkpointer?: Checkpointer;
  progress: TrainingProgress;
} {
  const earlyStopping = options.earlyStopping
    ? new EarlyStopping(options.earlyStopping)
    : undefined;

  const lrScheduler = options.lrScheduler
    ? new LRScheduler(options.lrScheduler)
    : undefined;

  const checkpointer = options.checkpointing
    ? new Checkpointer(options.checkpointing)
    : undefined;

  const progress = new TrainingProgress();

  const callbacks: TrainingCallbacks = {
    onTrainBegin: () => {
      progress.start();
      if (options.verbose) {
        console.info('Training started...');
      }
    },

    onEpochEnd: async (epoch, logs) => {
      progress.recordEpoch(logs);

      // Check early stopping
      if (earlyStopping?.check(epoch, logs)) {
        // Training should stop - handled by trainer
      }
    },

    onTrainEnd: (logs) => {
      const summary = progress.getSummary();
      if (options.verbose) {
        console.info('\n=== Training Complete ===');
        console.info(`Epochs completed: ${logs.epochs}`);
        console.info(`Total batches: ${logs.totalBatches}`);
        console.info(`Final loss: ${logs.finalLoss.toFixed(6)}`);
        if (logs.finalValLoss !== undefined) {
          console.info(`Final val_loss: ${logs.finalValLoss.toFixed(6)}`);
        }
        if (logs.stoppedEarly) {
          console.info('Training stopped early due to early stopping criteria');
        }
        console.info(`Total time: ${TrainingProgress.formatTime(summary.totalTime)}`);
        console.info(`Best loss: ${summary.bestLoss.toFixed(6)}`);
        if (summary.bestValLoss !== undefined) {
          console.info(`Best val_loss: ${summary.bestValLoss.toFixed(6)}`);
        }
      }
    },
  };

  return { callbacks, earlyStopping, lrScheduler, checkpointer, progress };
}

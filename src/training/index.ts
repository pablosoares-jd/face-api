/**
 * Face-API Training Module
 *
 * Provides tools for training and fine-tuning face recognition models.
 *
 * This module is separate from the core face-api to keep bundle sizes small.
 * Import only when you need training capabilities.
 *
 * @example
 * ```typescript
 * // Import training module
 * import * as training from '@vladmandic/face-api/training';
 *
 * // Or import specific classes
 * import { FaceClassifier, ModelTrainer, DatasetLoader } from '@vladmandic/face-api/training';
 * ```
 *
 * @module training
 */

// KNN-based Face Classifier
// Re-export tf for convenience (direct import to avoid circular dependency)
import * as tf from '@tensorflow/tfjs';

export { FaceClassifier } from './FaceClassifier';
export type {
  LabeledFaceDescriptor,
  ClassificationResult,
  ClassifyOptions,
  SerializedClassifier,
  DistanceMetric,
} from './FaceClassifier';

// Model Trainer for fine-tuning
export { ModelTrainer, tripletLoss, contrastiveLoss } from './ModelTrainer';
export type {
  TrainingConfig,
  TrainingCallbacks,
  TrainingLogs,
  EpochLogs,
  BatchLogs,
} from './ModelTrainer';

// Dataset Loader
export { DatasetLoader, TrainingDataset } from './DatasetLoader';
export type {
  TrainingSample,
  DatasetMetadata,
  TrainingTask,
  DirectoryOptions,
  CSVOptions,
  TensorDatasetOptions,
} from './DatasetLoader';

// Image Augmentation
export { ImageAugmenter, AugmentationPipeline, AugmentationPresets } from './ImageAugmenter';
export type {
  AugmentationConfig,
  AugmentationResult,
  AugmentRange,
} from './ImageAugmenter';

// Training Utilities
export {
  EarlyStopping,
  LRScheduler,
  Checkpointer,
  clipGradients,
  createAdamW,
  createCallbacks,
} from './TrainingUtils';
export type {
  EarlyStoppingConfig,
  LRSchedulerConfig,
  ScheduleType,
  CheckpointConfig,
  GradientClipConfig,
  AdamWConfig,
  TrainingProgress,
} from './TrainingUtils';

// Specialized Trainers for specific models
export {
  AgeGenderTrainer,
  ExpressionTrainer,
  EXPRESSIONS,
  EmbeddingTrainer,
  createAgeGenderLabels,
  createExpressionLabels,
  PUBLIC_DATASETS,
} from './SpecializedTrainers';
export type { AgeGenderLabels, Expression } from './SpecializedTrainers';

// Model Analysis and Baseline
export { ModelAnalyzer } from './ModelAnalyzer';
export type {
  ParamStats,
  ModelAnalysis,
  BaselineMetrics,
  ModelComparison,
  QuickAnalysis,
} from './ModelAnalyzer';
export { tf };

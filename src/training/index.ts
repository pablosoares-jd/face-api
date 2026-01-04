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
export {
  FaceClassifier,
  LabeledFaceDescriptor,
  ClassificationResult,
  ClassifyOptions,
  SerializedClassifier,
  DistanceMetric
} from './FaceClassifier';

// Model Trainer for fine-tuning
export {
  ModelTrainer,
  TrainingConfig,
  TrainingCallbacks,
  TrainingLogs,
  EpochLogs,
  BatchLogs,
  tripletLoss,
  contrastiveLoss
} from './ModelTrainer';

// Dataset Loader
export {
  DatasetLoader,
  TrainingDataset,
  TrainingSample,
  DatasetMetadata,
  TrainingTask,
  DirectoryOptions,
  CSVOptions,
  TensorDatasetOptions
} from './DatasetLoader';

// Image Augmentation
export {
  ImageAugmenter,
  AugmentationPipeline,
  AugmentationPresets,
  AugmentationConfig,
  AugmentationResult,
  AugmentRange
} from './ImageAugmenter';

// Training Utilities
export {
  EarlyStopping,
  EarlyStoppingConfig,
  LRScheduler,
  LRSchedulerConfig,
  ScheduleType,
  Checkpointer,
  CheckpointConfig,
  GradientClipConfig,
  clipGradients,
  AdamWConfig,
  createAdamW,
  TrainingProgress,
  createCallbacks
} from './TrainingUtils';

// Specialized Trainers for specific models
export {
  AgeGenderTrainer,
  AgeGenderLabels,
  ExpressionTrainer,
  Expression,
  EXPRESSIONS,
  EmbeddingTrainer,
  createAgeGenderLabels,
  createExpressionLabels,
  PUBLIC_DATASETS
} from './SpecializedTrainers';

// Re-export tf for convenience (direct import to avoid circular dependency)
import * as tf from '@tensorflow/tfjs';
export { tf };

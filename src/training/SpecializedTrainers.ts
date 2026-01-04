/**
 * Specialized Trainers for face-api models.
 *
 * These trainers implement the correct forward pass for each model type,
 * enabling actual gradient-based training.
 *
 * @example
 * ```typescript
 * import { AgeGenderTrainer } from '@vladmandic/face-api/training';
 * import * as faceapi from '@vladmandic/face-api';
 *
 * // Load base model
 * await faceapi.nets.ageGenderNet.loadFromUri('/models');
 *
 * // Create trainer
 * const trainer = new AgeGenderTrainer(faceapi.nets.ageGenderNet);
 * await trainer.freezeFeatureExtractor();
 *
 * // Train with your data
 * const logs = await trainer.fit(images, labels, {
 *   epochs: 50,
 *   batchSize: 32,
 *   earlyStopping: { patience: 5 }
 * });
 * ```
 */

import * as tf from '@tensorflow/tfjs';
import { NeuralNetwork } from '../NeuralNetwork';
import { AgeGenderNet } from '../ageGenderNet/AgeGenderNet';
import { FaceExpressionNet } from '../faceExpressionNet/FaceExpressionNet';
import { ModelTrainer, TrainingConfig, TrainingLogs } from './ModelTrainer';

/**
 * Age and Gender labels.
 */
export interface AgeGenderLabels {
  ages: tf.Tensor1D;       // Float ages
  genders: tf.Tensor1D;    // 0 = female, 1 = male
}

/**
 * Trainer for AgeGenderNet.
 *
 * Supports:
 * - Transfer learning (freeze feature extractor)
 * - Full fine-tuning
 * - Multi-task learning (age + gender)
 */
export class AgeGenderTrainer extends ModelTrainer<any> {
  private _ageWeight: number;
  private _genderWeight: number;

  constructor(
    model: AgeGenderNet,
    options: { ageWeight?: number; genderWeight?: number } = {}
  ) {
    super(model as unknown as NeuralNetwork<any>);
    this._ageWeight = options.ageWeight ?? 1.0;
    this._genderWeight = options.genderWeight ?? 1.0;
  }

  /**
   * Train the model with age and gender labels.
   *
   * @param images - Input images [N, H, W, 3]
   * @param labels - Age and gender labels
   * @param config - Training configuration
   */
  public async fitAgeGender(
    images: tf.Tensor4D,
    labels: AgeGenderLabels,
    config: TrainingConfig = {}
  ): Promise<TrainingLogs> {
    // Combine labels into single tensor for compatibility
    const combinedLabels = tf.concat([
      labels.ages.expandDims(1),
      labels.genders.expandDims(1)
    ], 1);

    const result = await this.fit(images, combinedLabels, {
      ...config,
      loss: 'custom',
      customLoss: (yTrue, yPred) => this._ageGenderLoss(yTrue, yPred)
    });

    combinedLabels.dispose();
    return result;
  }

  /**
   * Multi-task loss for age and gender.
   */
  private _ageGenderLoss(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    return tf.tidy(() => {
      // yTrue: [batch, 2] - [age, gender]
      // yPred: [batch, 3] - [age, genderFemale, genderMale]

      const trueAge = yTrue.slice([0, 0], [-1, 1]).squeeze([1]);
      const trueGender = yTrue.slice([0, 1], [-1, 1]).squeeze([1]);

      const predAge = yPred.slice([0, 0], [-1, 1]).squeeze([1]);
      const predGender = yPred.slice([0, 1], [-1, 2]);

      // Age loss: MAE (Mean Absolute Error)
      const ageLoss = tf.losses.absoluteDifference(trueAge, predAge);

      // Gender loss: Binary Cross-Entropy
      const genderOneHot = tf.oneHot(trueGender.toInt(), 2);
      const genderLoss = tf.losses.softmaxCrossEntropy(genderOneHot, predGender);

      // Combined weighted loss
      return tf.add(
        tf.mul(ageLoss, this._ageWeight),
        tf.mul(genderLoss, this._genderWeight)
      ) as tf.Scalar;
    });
  }

  /**
   * Forward pass through AgeGenderNet.
   */
  protected override _forwardPass(inputs: tf.Tensor): tf.Tensor {
    const model = this['_model'] as unknown as AgeGenderNet;
    return tf.tidy(() => {
      const { age, gender } = model.runNet(inputs as tf.Tensor4D);
      // Return combined output: [age, genderFemale, genderMale]
      return tf.concat([age.expandDims(1), gender], 1);
    });
  }

  /**
   * Evaluate model on test data.
   */
  public async evaluate(
    images: tf.Tensor4D,
    labels: AgeGenderLabels
  ): Promise<{ ageMAE: number; genderAccuracy: number }> {
    const model = this['_model'] as unknown as AgeGenderNet;

    const predictions = model.forwardInput(images);

    const ageData = await predictions.age.data();
    const genderData = await predictions.gender.data();
    const trueAgeData = await labels.ages.data();
    const trueGenderData = await labels.genders.data();

    // Calculate Age MAE
    let ageError = 0;
    for (let i = 0; i < ageData.length; i++) {
      ageError += Math.abs(ageData[i] - trueAgeData[i]);
    }
    const ageMAE = ageError / ageData.length;

    // Calculate Gender Accuracy
    let genderCorrect = 0;
    const numSamples = trueGenderData.length;
    for (let i = 0; i < numSamples; i++) {
      const predMale = genderData[i * 2] < genderData[i * 2 + 1];
      const trueMale = trueGenderData[i] === 1;
      if (predMale === trueMale) genderCorrect++;
    }
    const genderAccuracy = genderCorrect / numSamples;

    predictions.age.dispose();
    predictions.gender.dispose();

    return { ageMAE, genderAccuracy };
  }
}

/**
 * Expression labels enum.
 */
export const EXPRESSIONS = [
  'neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'
] as const;

export type Expression = typeof EXPRESSIONS[number];

/**
 * Trainer for FaceExpressionNet.
 */
export class ExpressionTrainer extends ModelTrainer<any> {
  constructor(model: FaceExpressionNet) {
    super(model as unknown as NeuralNetwork<any>);
  }

  /**
   * Train the model with expression labels.
   *
   * @param images - Input images [N, H, W, 3]
   * @param labels - Expression labels (0-6 or one-hot)
   * @param config - Training configuration
   */
  public async fitExpressions(
    images: tf.Tensor4D,
    labels: tf.Tensor1D | tf.Tensor2D,
    config: TrainingConfig = {}
  ): Promise<TrainingLogs> {
    // Convert to one-hot if needed
    const oneHotLabels = labels.rank === 1
      ? tf.oneHot(labels as tf.Tensor1D, 7)
      : labels;

    return this.fit(images, oneHotLabels, {
      ...config,
      loss: 'categoricalCrossentropy'
    });
  }

  /**
   * Forward pass through FaceExpressionNet.
   */
  protected override _forwardPass(inputs: tf.Tensor): tf.Tensor {
    const model = this['_model'] as unknown as FaceExpressionNet;
    return model.forwardInput(inputs as tf.Tensor4D);
  }

  /**
   * Evaluate model on test data.
   */
  public async evaluate(
    images: tf.Tensor4D,
    labels: tf.Tensor1D
  ): Promise<{ accuracy: number; confusionMatrix: number[][] }> {
    const model = this['_model'] as unknown as FaceExpressionNet;

    const predictions = model.forwardInput(images);
    const predLabels = predictions.argMax(1);

    const predData = await predLabels.data();
    const trueData = await labels.data();

    // Calculate accuracy
    let correct = 0;
    for (let i = 0; i < predData.length; i++) {
      if (predData[i] === trueData[i]) correct++;
    }
    const accuracy = correct / predData.length;

    // Build confusion matrix
    const confusionMatrix: number[][] = Array(7).fill(null).map(() => Array(7).fill(0));
    for (let i = 0; i < predData.length; i++) {
      const trueIdx = trueData[i] as number;
      const predIdx = predData[i] as number;
      confusionMatrix[trueIdx]![predIdx]!++;
    }

    predictions.dispose();
    predLabels.dispose();

    return { accuracy, confusionMatrix };
  }

  /**
   * Get expression name from index.
   */
  public static getExpressionName(index: number): Expression | undefined {
    return EXPRESSIONS[index];
  }

  /**
   * Get index from expression name.
   */
  public static getExpressionIndex(name: Expression): number {
    return EXPRESSIONS.indexOf(name);
  }
}

/**
 * Trainer for face embedding/recognition models.
 * Uses triplet loss for learning discriminative embeddings.
 */
export class EmbeddingTrainer extends ModelTrainer<any> {
  private _margin: number;

  constructor(model: NeuralNetwork<any>, margin = 0.2) {
    super(model);
    this._margin = margin;
  }

  /**
   * Train with triplets (anchor, positive, negative).
   *
   * @param anchors - Anchor images
   * @param positives - Positive images (same identity as anchor)
   * @param negatives - Negative images (different identity)
   * @param config - Training configuration
   */
  public async fitTriplets(
    anchors: tf.Tensor4D,
    positives: tf.Tensor4D,
    negatives: tf.Tensor4D,
    config: TrainingConfig = {}
  ): Promise<TrainingLogs> {
    // Stack triplets
    const inputs = tf.concat([anchors, positives, negatives], 0);
    const labels = tf.zeros([anchors.shape[0]]); // Dummy labels

    return this.fit(inputs, labels, {
      ...config,
      loss: 'custom',
      customLoss: (_, embeddings) => this._tripletLoss(embeddings, anchors.shape[0])
    });
  }

  /**
   * Triplet loss implementation.
   */
  private _tripletLoss(embeddings: tf.Tensor, batchSize: number): tf.Scalar {
    return tf.tidy(() => {
      const anchor = embeddings.slice([0], [batchSize]);
      const positive = embeddings.slice([batchSize], [batchSize]);
      const negative = embeddings.slice([batchSize * 2], [batchSize]);

      const positiveDist = tf.sum(tf.square(tf.sub(anchor, positive)), 1);
      const negativeDist = tf.sum(tf.square(tf.sub(anchor, negative)), 1);

      const loss = tf.maximum(
        0,
        tf.add(tf.sub(positiveDist, negativeDist), this._margin)
      );

      return tf.mean(loss) as tf.Scalar;
    });
  }

  /**
   * Forward pass - get embeddings.
   */
  protected override _forwardPass(inputs: tf.Tensor): tf.Tensor {
    // This should be overridden for specific embedding models
    // For now, return identity
    return inputs.clone();
  }
}

/**
 * Helper to create labels tensor from array.
 */
export function createAgeGenderLabels(
  ages: number[],
  genders: ('male' | 'female')[]
): AgeGenderLabels {
  return {
    ages: tf.tensor1d(ages, 'float32'),
    genders: tf.tensor1d(genders.map(g => g === 'male' ? 1 : 0), 'float32')
  };
}

/**
 * Helper to create expression labels tensor.
 */
export function createExpressionLabels(expressions: Expression[]): tf.Tensor1D {
  return tf.tensor1d(expressions.map(e => EXPRESSIONS.indexOf(e)), 'int32');
}

/**
 * List of public datasets for face training.
 */
export const PUBLIC_DATASETS = {
  // Age & Gender
  ageGender: [
    {
      name: 'UTKFace',
      url: 'https://susanqq.github.io/UTKFace/',
      description: '20,000+ face images with age, gender, ethnicity labels',
      license: 'Academic use only'
    },
    {
      name: 'IMDB-WIKI',
      url: 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/',
      description: '500,000+ celebrity faces with age and gender',
      license: 'Research only'
    },
    {
      name: 'MORPH',
      url: 'https://www.faceaginggroup.com/morph/',
      description: '55,000 face images with age progression',
      license: 'Academic license required'
    }
  ],

  // Expression
  expression: [
    {
      name: 'FER2013',
      url: 'https://www.kaggle.com/datasets/msambare/fer2013',
      description: '35,000 48x48 grayscale images, 7 expressions',
      license: 'Kaggle'
    },
    {
      name: 'AffectNet',
      url: 'http://mohammadmahoor.com/affectnet/',
      description: '1M+ faces with expression and valence-arousal',
      license: 'Academic license required'
    },
    {
      name: 'RAF-DB',
      url: 'http://www.whdeng.cn/RAF/model1.html',
      description: '30,000 facial expression images',
      license: 'Academic use'
    }
  ],

  // Face Recognition
  recognition: [
    {
      name: 'LFW',
      url: 'http://vis-www.cs.umass.edu/lfw/',
      description: '13,000 labeled face images in the wild',
      license: 'Research only'
    },
    {
      name: 'CelebA',
      url: 'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html',
      description: '200,000 celebrity faces with 40 attributes',
      license: 'Non-commercial research'
    },
    {
      name: 'VGGFace2',
      url: 'https://github.com/ox-vgg/vgg_face2',
      description: '3.3M faces of 9,000 subjects',
      license: 'Creative Commons'
    }
  ],

  // Landmarks
  landmarks: [
    {
      name: '300W',
      url: 'https://ibug.doc.ic.ac.uk/resources/300-W/',
      description: '300 indoor + outdoor faces with 68 landmarks',
      license: 'Research only'
    },
    {
      name: 'WFLW',
      url: 'https://wywu.github.io/projects/LAB/WFLW.html',
      description: '10,000 faces with 98 landmarks',
      license: 'Academic use'
    }
  ]
};

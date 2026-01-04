/**
 * Model Analyzer for baseline evaluation and model comparison.
 *
 * Provides tools to analyze pre-trained models, establish baselines,
 * and compare performance after fine-tuning.
 *
 * @module training/ModelAnalyzer
 */

import * as tf from '@tensorflow/tfjs';
import { NeuralNetwork } from '../NeuralNetwork';

/**
 * Statistics for a single parameter tensor.
 */
export interface ParamStats {
  path: string;
  shape: number[];
  numParams: number;
  dtype: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  sparsity: number; // percentage of near-zero values
  isTrainable: boolean;
}

/**
 * Overall model analysis result.
 */
export interface ModelAnalysis {
  name: string;
  totalParams: number;
  trainableParams: number;
  frozenParams: number;
  layers: ParamStats[];
  memoryBytes: number;
  memoryMB: number;
  summary: string;
}

/**
 * Baseline metrics for a model.
 */
export interface BaselineMetrics {
  modelName: string;
  timestamp: string;
  testSamples: number;
  metrics: Record<string, number>;
  perClassMetrics?: Record<string, Record<string, number>>;
  confusionMatrix?: number[][];
  latency: {
    mean: number;
    std: number;
    min: number;
    max: number;
    p50: number;
    p95: number;
    p99: number;
  };
}

/**
 * Comparison between two model states.
 */
export interface ModelComparison {
  baseline: BaselineMetrics;
  current: BaselineMetrics;
  improvements: Record<string, number>; // percentage change
  paramChanges: {
    path: string;
    meanDiff: number;
    maxDiff: number;
    l2Distance: number;
  }[];
}

/**
 * Model Analyzer class for analyzing neural networks.
 */
export class ModelAnalyzer {
  /**
   * Analyze a neural network's architecture and parameters.
   */
  public static async analyze<T>(model: NeuralNetwork<T>): Promise<ModelAnalysis> {
    if (!model.isLoaded) {
      throw new Error('Model must be loaded before analysis');
    }

    const paramList = model.getParamList();
    const trainableParams = model.getTrainableParams();
    const trainablePaths = new Set(trainableParams.map(p => p.path));

    const layers: ParamStats[] = [];
    let totalParams = 0;
    let memoryBytes = 0;

    for (const { path, tensor } of paramList) {
      const stats = await this.analyzeParam(path, tensor, trainablePaths.has(path));
      layers.push(stats);
      totalParams += stats.numParams;
      memoryBytes += stats.numParams * this.dtypeBytes(stats.dtype);
    }

    const trainableCount = layers
      .filter(l => l.isTrainable)
      .reduce((sum, l) => sum + l.numParams, 0);

    const summary = this.generateSummary(model._name, layers, totalParams);

    return {
      name: model._name,
      totalParams,
      trainableParams: trainableCount,
      frozenParams: totalParams - trainableCount,
      layers,
      memoryBytes,
      memoryMB: memoryBytes / (1024 * 1024),
      summary
    };
  }

  /**
   * Analyze a single parameter tensor.
   */
  private static async analyzeParam(
    path: string,
    tensor: tf.Tensor,
    isTrainable: boolean
  ): Promise<ParamStats> {
    const data = await tensor.data();
    const numParams = data.length;

    // Calculate statistics
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let nearZero = 0;
    const threshold = 1e-6;

    for (let i = 0; i < numParams; i++) {
      const val = data[i];
      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
      if (Math.abs(val) < threshold) nearZero++;
    }

    const mean = sum / numParams;

    // Calculate std
    let sumSqDiff = 0;
    for (let i = 0; i < numParams; i++) {
      sumSqDiff += (data[i] - mean) ** 2;
    }
    const std = Math.sqrt(sumSqDiff / numParams);

    return {
      path,
      shape: tensor.shape,
      numParams,
      dtype: tensor.dtype,
      min,
      max,
      mean,
      std,
      sparsity: (nearZero / numParams) * 100,
      isTrainable
    };
  }

  /**
   * Get bytes per element for a dtype.
   */
  private static dtypeBytes(dtype: string): number {
    switch (dtype) {
      case 'float32': return 4;
      case 'int32': return 4;
      case 'float16': return 2;
      case 'int16': return 2;
      case 'int8': return 1;
      case 'bool': return 1;
      default: return 4;
    }
  }

  /**
   * Generate human-readable summary.
   */
  private static generateSummary(
    name: string,
    layers: ParamStats[],
    totalParams: number
  ): string {
    const lines: string[] = [
      `Model: ${name}`,
      `${'='.repeat(60)}`,
      `Total Parameters: ${totalParams.toLocaleString()}`,
      ``,
      `Layer Analysis:`,
      `-`.repeat(60),
    ];

    // Group by layer type
    const layerGroups = new Map<string, { count: number; params: number }>();
    for (const layer of layers) {
      const layerType = this.inferLayerType(layer.path, layer.shape);
      const existing = layerGroups.get(layerType) || { count: 0, params: 0 };
      layerGroups.set(layerType, {
        count: existing.count + 1,
        params: existing.params + layer.numParams
      });
    }

    for (const [type, stats] of layerGroups) {
      const pct = ((stats.params / totalParams) * 100).toFixed(1);
      lines.push(`  ${type}: ${stats.count} tensors, ${stats.params.toLocaleString()} params (${pct}%)`);
    }

    lines.push(``);
    lines.push(`Weight Statistics:`);
    lines.push(`-`.repeat(60));

    // Overall weight stats
    const allMeans = layers.map(l => l.mean);
    const allStds = layers.map(l => l.std);
    const avgMean = allMeans.reduce((a, b) => a + b, 0) / allMeans.length;
    const avgStd = allStds.reduce((a, b) => a + b, 0) / allStds.length;
    const avgSparsity = layers.reduce((a, l) => a + l.sparsity, 0) / layers.length;

    lines.push(`  Average Mean: ${avgMean.toExponential(3)}`);
    lines.push(`  Average Std: ${avgStd.toExponential(3)}`);
    lines.push(`  Average Sparsity: ${avgSparsity.toFixed(2)}%`);

    return lines.join('\n');
  }

  /**
   * Infer layer type from path and shape.
   */
  private static inferLayerType(path: string, shape: number[]): string {
    const pathLower = path.toLowerCase();

    if (pathLower.includes('bias')) return 'Bias';
    if (pathLower.includes('bn') || pathLower.includes('batchnorm')) {
      if (pathLower.includes('mean')) return 'BatchNorm Mean';
      if (pathLower.includes('variance')) return 'BatchNorm Variance';
      if (pathLower.includes('scale') || pathLower.includes('gamma')) return 'BatchNorm Scale';
      if (pathLower.includes('offset') || pathLower.includes('beta')) return 'BatchNorm Offset';
      return 'BatchNorm';
    }
    if (shape.length === 4) return 'Conv2D';
    if (shape.length === 2) return 'Dense';
    if (shape.length === 1) return 'Vector';

    return 'Other';
  }

  /**
   * Benchmark model latency.
   */
  public static async benchmarkLatency<T>(
    model: NeuralNetwork<T>,
    createInput: () => tf.Tensor,
    forwardFn: (input: tf.Tensor) => tf.Tensor | Promise<tf.Tensor>,
    options: {
      warmupRuns?: number;
      benchmarkRuns?: number;
    } = {}
  ): Promise<BaselineMetrics['latency']> {
    const { warmupRuns = 5, benchmarkRuns = 50 } = options;

    // Warmup
    for (let i = 0; i < warmupRuns; i++) {
      const input = createInput();
      const output = await forwardFn(input);
      if (output instanceof tf.Tensor) output.dispose();
      input.dispose();
    }

    // Benchmark
    const times: number[] = [];
    for (let i = 0; i < benchmarkRuns; i++) {
      const input = createInput();

      const start = performance.now();
      const output = await forwardFn(input);
      await tf.ready(); // Ensure GPU sync
      const end = performance.now();

      times.push(end - start);

      if (output instanceof tf.Tensor) output.dispose();
      input.dispose();
    }

    // Calculate statistics
    times.sort((a, b) => a - b);
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const variance = times.reduce((a, t) => a + (t - mean) ** 2, 0) / times.length;
    const std = Math.sqrt(variance);

    return {
      mean,
      std,
      min: times[0],
      max: times[times.length - 1],
      p50: times[Math.floor(times.length * 0.5)],
      p95: times[Math.floor(times.length * 0.95)],
      p99: times[Math.floor(times.length * 0.99)]
    };
  }

  /**
   * Evaluate model on a test dataset and create baseline metrics.
   */
  public static async createBaseline<T>(
    model: NeuralNetwork<T>,
    testData: { inputs: tf.Tensor; labels: tf.Tensor },
    forwardFn: (input: tf.Tensor) => tf.Tensor,
    options: {
      taskType?: 'classification' | 'regression' | 'multiLabel';
      classNames?: string[];
      batchSize?: number;
    } = {}
  ): Promise<BaselineMetrics> {
    const {
      taskType = 'classification',
      classNames,
      batchSize = 32
    } = options;

    const numSamples = testData.inputs.shape[0];
    const metrics: Record<string, number> = {};
    let confusionMatrix: number[][] | undefined;
    let perClassMetrics: Record<string, Record<string, number>> | undefined;

    // Run predictions in batches
    const predictions: number[][] = [];
    const labels: number[][] = [];

    for (let i = 0; i < numSamples; i += batchSize) {
      const end = Math.min(i + batchSize, numSamples);
      const batchInputs = testData.inputs.slice(i, end);
      const batchLabels = testData.labels.slice(i, end);

      const batchPreds = tf.tidy(() => forwardFn(batchInputs));

      const [predsData, labelsData] = await Promise.all([
        batchPreds.array() as Promise<number[][]>,
        batchLabels.array() as Promise<number[][]>
      ]);

      predictions.push(...predsData);
      labels.push(...labelsData);

      batchPreds.dispose();
      batchInputs.dispose();
      batchLabels.dispose();
    }

    if (taskType === 'classification') {
      const result = this.calculateClassificationMetrics(predictions, labels, classNames);
      Object.assign(metrics, result.metrics);
      confusionMatrix = result.confusionMatrix;
      perClassMetrics = result.perClassMetrics;
    } else if (taskType === 'regression') {
      Object.assign(metrics, this.calculateRegressionMetrics(predictions, labels));
    }

    // Benchmark latency
    const latency = await this.benchmarkLatency(
      model,
      () => tf.randomNormal([1, ...testData.inputs.shape.slice(1)]),
      forwardFn
    );

    return {
      modelName: model._name,
      timestamp: new Date().toISOString(),
      testSamples: numSamples,
      metrics,
      perClassMetrics,
      confusionMatrix,
      latency
    };
  }

  /**
   * Calculate classification metrics.
   */
  private static calculateClassificationMetrics(
    predictions: number[][],
    labels: number[][],
    classNames?: string[]
  ): {
    metrics: Record<string, number>;
    confusionMatrix: number[][];
    perClassMetrics: Record<string, Record<string, number>>;
  } {
    const numClasses = predictions[0].length;
    const numSamples = predictions.length;

    // Convert to class indices
    const predIndices = predictions.map(p => p.indexOf(Math.max(...p)));
    const trueIndices = labels.map(l => l.indexOf(Math.max(...l)));

    // Build confusion matrix
    const confusionMatrix: number[][] = Array(numClasses)
      .fill(null)
      .map(() => Array(numClasses).fill(0));

    let correct = 0;
    for (let i = 0; i < numSamples; i++) {
      confusionMatrix[trueIndices[i]][predIndices[i]]++;
      if (trueIndices[i] === predIndices[i]) correct++;
    }

    const accuracy = correct / numSamples;

    // Per-class metrics
    const perClassMetrics: Record<string, Record<string, number>> = {};

    for (let c = 0; c < numClasses; c++) {
      const className = classNames?.[c] ?? `class_${c}`;
      const tp = confusionMatrix[c][c];
      const fp = confusionMatrix.reduce((sum, row, i) => i !== c ? sum + row[c] : sum, 0);
      const fn = confusionMatrix[c].reduce((sum, val, i) => i !== c ? sum + val : sum, 0);
      const tn = numSamples - tp - fp - fn;

      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
      const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;

      perClassMetrics[className] = {
        precision,
        recall,
        f1,
        specificity,
        support: tp + fn
      };
    }

    // Macro averages
    const classes = Object.values(perClassMetrics);
    const macroPrecision = classes.reduce((s, c) => s + c.precision, 0) / numClasses;
    const macroRecall = classes.reduce((s, c) => s + c.recall, 0) / numClasses;
    const macroF1 = classes.reduce((s, c) => s + c.f1, 0) / numClasses;

    // Weighted averages (by support)
    const totalSupport = classes.reduce((s, c) => s + c.support, 0);
    const weightedPrecision = classes.reduce((s, c) => s + c.precision * c.support, 0) / totalSupport;
    const weightedRecall = classes.reduce((s, c) => s + c.recall * c.support, 0) / totalSupport;
    const weightedF1 = classes.reduce((s, c) => s + c.f1 * c.support, 0) / totalSupport;

    return {
      metrics: {
        accuracy,
        macroPrecision,
        macroRecall,
        macroF1,
        weightedPrecision,
        weightedRecall,
        weightedF1
      },
      confusionMatrix,
      perClassMetrics
    };
  }

  /**
   * Calculate regression metrics.
   */
  private static calculateRegressionMetrics(
    predictions: number[][],
    labels: number[][]
  ): Record<string, number> {
    const numSamples = predictions.length;
    const numOutputs = predictions[0].length;

    let mse = 0;
    let mae = 0;
    let ssRes = 0;
    let ssTot = 0;

    // Calculate mean for R²
    const labelMeans = Array(numOutputs).fill(0);
    for (const label of labels) {
      for (let j = 0; j < numOutputs; j++) {
        labelMeans[j] += label[j];
      }
    }
    for (let j = 0; j < numOutputs; j++) {
      labelMeans[j] /= numSamples;
    }

    for (let i = 0; i < numSamples; i++) {
      for (let j = 0; j < numOutputs; j++) {
        const diff = predictions[i][j] - labels[i][j];
        mse += diff * diff;
        mae += Math.abs(diff);
        ssRes += diff * diff;
        ssTot += (labels[i][j] - labelMeans[j]) ** 2;
      }
    }

    const n = numSamples * numOutputs;
    mse /= n;
    mae /= n;
    const rmse = Math.sqrt(mse);
    const r2 = 1 - ssRes / ssTot;

    return { mse, rmse, mae, r2 };
  }

  /**
   * Compare two baselines or model states.
   */
  public static compareBaselines(
    baseline: BaselineMetrics,
    current: BaselineMetrics
  ): ModelComparison {
    const improvements: Record<string, number> = {};

    // Compare metrics
    for (const [key, baseVal] of Object.entries(baseline.metrics)) {
      const currVal = current.metrics[key];
      if (currVal !== undefined && baseVal !== 0) {
        improvements[key] = ((currVal - baseVal) / Math.abs(baseVal)) * 100;
      }
    }

    // Latency improvement (negative is better)
    improvements['latency_mean'] = ((current.latency.mean - baseline.latency.mean) / baseline.latency.mean) * 100;

    return {
      baseline,
      current,
      improvements,
      paramChanges: [] // Filled when comparing model weights
    };
  }

  /**
   * Compare parameter changes between two model states.
   */
  public static async compareParams<T>(
    modelA: NeuralNetwork<T>,
    modelB: NeuralNetwork<T>
  ): Promise<ModelComparison['paramChanges']> {
    const paramsA = modelA.getParamList();
    const paramsB = modelB.getParamList();

    const changes: ModelComparison['paramChanges'] = [];

    for (let i = 0; i < paramsA.length; i++) {
      const pathA = paramsA[i].path;
      const tensorA = paramsA[i].tensor;

      // Find matching param in B
      const paramB = paramsB.find(p => p.path === pathA);
      if (!paramB) continue;

      const [dataA, dataB] = await Promise.all([
        tensorA.data(),
        paramB.tensor.data()
      ]);

      let maxDiff = 0;
      let sumDiff = 0;
      let sumSqDiff = 0;

      for (let j = 0; j < dataA.length; j++) {
        const diff = Math.abs(dataA[j] - dataB[j]);
        maxDiff = Math.max(maxDiff, diff);
        sumDiff += dataA[j] - dataB[j];
        sumSqDiff += (dataA[j] - dataB[j]) ** 2;
      }

      const meanDiff = sumDiff / dataA.length;
      const l2Distance = Math.sqrt(sumSqDiff);

      changes.push({
        path: pathA,
        meanDiff,
        maxDiff,
        l2Distance
      });
    }

    return changes;
  }

  /**
   * Export baseline to JSON.
   */
  public static exportBaseline(baseline: BaselineMetrics): string {
    return JSON.stringify(baseline, null, 2);
  }

  /**
   * Import baseline from JSON.
   */
  public static importBaseline(json: string): BaselineMetrics {
    return JSON.parse(json) as BaselineMetrics;
  }

  /**
   * Generate a human-readable report from baseline metrics.
   */
  public static generateReport(baseline: BaselineMetrics): string {
    const lines: string[] = [
      `Baseline Report: ${baseline.modelName}`,
      `${'='.repeat(60)}`,
      `Timestamp: ${baseline.timestamp}`,
      `Test Samples: ${baseline.testSamples}`,
      ``,
      `Metrics:`,
      `-`.repeat(40)
    ];

    for (const [key, value] of Object.entries(baseline.metrics)) {
      lines.push(`  ${key}: ${(value * 100).toFixed(2)}%`);
    }

    lines.push(``);
    lines.push(`Latency (ms):`);
    lines.push(`-`.repeat(40));
    lines.push(`  Mean: ${baseline.latency.mean.toFixed(2)}`);
    lines.push(`  Std: ${baseline.latency.std.toFixed(2)}`);
    lines.push(`  P50: ${baseline.latency.p50.toFixed(2)}`);
    lines.push(`  P95: ${baseline.latency.p95.toFixed(2)}`);
    lines.push(`  P99: ${baseline.latency.p99.toFixed(2)}`);

    if (baseline.perClassMetrics) {
      lines.push(``);
      lines.push(`Per-Class Metrics:`);
      lines.push(`-`.repeat(40));
      for (const [className, metrics] of Object.entries(baseline.perClassMetrics)) {
        lines.push(`  ${className}:`);
        lines.push(`    Precision: ${(metrics.precision * 100).toFixed(2)}%`);
        lines.push(`    Recall: ${(metrics.recall * 100).toFixed(2)}%`);
        lines.push(`    F1: ${(metrics.f1 * 100).toFixed(2)}%`);
        lines.push(`    Support: ${metrics.support}`);
      }
    }

    if (baseline.confusionMatrix) {
      lines.push(``);
      lines.push(`Confusion Matrix:`);
      lines.push(`-`.repeat(40));
      for (const row of baseline.confusionMatrix) {
        lines.push(`  [${row.map(v => v.toString().padStart(4)).join(', ')}]`);
      }
    }

    return lines.join('\n');
  }

  /**
   * Generate comparison report between two baselines.
   */
  public static generateComparisonReport(comparison: ModelComparison): string {
    const lines: string[] = [
      `Model Comparison Report`,
      `${'='.repeat(60)}`,
      `Baseline: ${comparison.baseline.modelName} (${comparison.baseline.timestamp})`,
      `Current: ${comparison.current.modelName} (${comparison.current.timestamp})`,
      ``,
      `Metric Improvements:`,
      `-`.repeat(40)
    ];

    for (const [key, improvement] of Object.entries(comparison.improvements)) {
      const sign = improvement > 0 ? '+' : '';
      const emoji = improvement > 0 ? '↑' : improvement < 0 ? '↓' : '→';
      lines.push(`  ${key}: ${sign}${improvement.toFixed(2)}% ${emoji}`);
    }

    if (comparison.paramChanges.length > 0) {
      lines.push(``);
      lines.push(`Parameter Changes (top 10 by L2 distance):`);
      lines.push(`-`.repeat(40));

      const sorted = [...comparison.paramChanges].sort((a, b) => b.l2Distance - a.l2Distance);
      for (const change of sorted.slice(0, 10)) {
        lines.push(`  ${change.path}: L2=${change.l2Distance.toExponential(3)}, maxDiff=${change.maxDiff.toExponential(3)}`);
      }
    }

    return lines.join('\n');
  }
}

/**
 * Quick analysis functions for common use cases.
 */
export const QuickAnalysis = {
  /**
   * Get a quick summary of model size.
   */
  async modelSize<T>(model: NeuralNetwork<T>): Promise<{
    params: number;
    memoryMB: number;
    trainable: number;
  }> {
    const analysis = await ModelAnalyzer.analyze(model);
    return {
      params: analysis.totalParams,
      memoryMB: analysis.memoryMB,
      trainable: analysis.trainableParams
    };
  },

  /**
   * Check if weights look healthy (no NaN, reasonable range).
   */
  async healthCheck<T>(model: NeuralNetwork<T>): Promise<{
    healthy: boolean;
    issues: string[];
  }> {
    const issues: string[] = [];
    const analysis = await ModelAnalyzer.analyze(model);

    for (const layer of analysis.layers) {
      if (isNaN(layer.mean) || isNaN(layer.std)) {
        issues.push(`${layer.path}: Contains NaN values`);
      }
      if (!isFinite(layer.max) || !isFinite(layer.min)) {
        issues.push(`${layer.path}: Contains Infinity values`);
      }
      if (layer.std === 0) {
        issues.push(`${layer.path}: Zero variance (dead weights)`);
      }
      if (layer.sparsity > 95) {
        issues.push(`${layer.path}: Highly sparse (${layer.sparsity.toFixed(1)}% near-zero)`);
      }
      if (Math.abs(layer.mean) > 10) {
        issues.push(`${layer.path}: Large mean (${layer.mean.toFixed(3)})`);
      }
      if (layer.std > 10) {
        issues.push(`${layer.path}: Large std (${layer.std.toFixed(3)})`);
      }
    }

    return {
      healthy: issues.length === 0,
      issues
    };
  },

  /**
   * Compare two model weight snapshots.
   */
  async weightDrift<T>(
    modelA: NeuralNetwork<T>,
    modelB: NeuralNetwork<T>
  ): Promise<{
    totalDrift: number;
    layerDrifts: { path: string; drift: number }[];
  }> {
    const changes = await ModelAnalyzer.compareParams(modelA, modelB);

    const layerDrifts = changes.map(c => ({
      path: c.path,
      drift: c.l2Distance
    }));

    const totalDrift = changes.reduce((sum, c) => sum + c.l2Distance, 0);

    return { totalDrift, layerDrifts };
  }
};

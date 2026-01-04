/**
 * Benchmark completo dos modelos face-api.
 * Usa dados sintéticos - não precisa de imagens reais.
 *
 * Executar:
 *   pnpm exec node examples/benchmark-models.js
 */

const tf = require('@tensorflow/tfjs');
const path = require('path');
const fs = require('fs');

// Configuração
const MODELS_PATH = path.join(__dirname, '../weights');
const OUTPUT_PATH = path.join(__dirname, '../baselines');
const WARMUP_RUNS = 3;
const BENCHMARK_RUNS = 20;

// Configuração dos modelos
const MODEL_CONFIGS = {
  'tiny_face_detector': {
    inputShape: [1, 416, 416, 3],
    description: 'Detector de faces leve'
  },
  'ssd_mobilenetv1': {
    inputShape: [1, 512, 512, 3],
    description: 'Detector SSD MobileNet'
  },
  'face_landmark_68_tiny': {
    inputShape: [1, 112, 112, 3],
    description: 'Landmarks 68 pontos (leve)'
  },
  'face_landmark_68': {
    inputShape: [1, 112, 112, 3],
    description: 'Landmarks 68 pontos'
  },
  'face_recognition': {
    inputShape: [1, 150, 150, 3],
    description: 'Embeddings 128D'
  },
  'face_expression': {
    inputShape: [1, 64, 64, 3],
    description: '7 expressões faciais'
  },
  'age_gender': {
    inputShape: [1, 112, 112, 3],
    description: 'Idade e gênero'
  }
};

// Batch sizes para testar
const BATCH_SIZES = [1, 2, 4, 8];

async function loadModelWeights(modelName) {
  const manifestPath = path.join(MODELS_PATH, `${modelName}_model-weights_manifest.json`);

  if (!fs.existsSync(manifestPath)) {
    return null;
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

  // Calcular info do modelo
  let totalParams = 0;
  let totalBytes = 0;

  for (const group of manifest) {
    for (const weight of group.weights) {
      const params = weight.shape.reduce((a, b) => a * b, 1);
      totalParams += params;
      totalBytes += params * 4;
    }
  }

  return {
    name: modelName,
    params: totalParams,
    memoryMB: totalBytes / 1024 / 1024,
    manifest
  };
}

function calculateStats(times) {
  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const variance = times.reduce((a, t) => a + (t - mean) ** 2, 0) / times.length;

  return {
    mean: mean,
    std: Math.sqrt(variance),
    min: times[0],
    max: times[times.length - 1],
    p50: times[Math.floor(times.length * 0.5)],
    p95: times[Math.floor(times.length * 0.95)],
    p99: times[Math.floor(times.length * 0.99)]
  };
}

async function benchmarkTensorOps() {
  console.log('\n📊 Benchmark de Operações TensorFlow.js\n');

  const results = {};

  // Matmul benchmark
  for (const size of [256, 512, 1024]) {
    const a = tf.randomNormal([size, size]);
    const b = tf.randomNormal([size, size]);

    // Warmup
    for (let i = 0; i < WARMUP_RUNS; i++) {
      tf.matMul(a, b).dispose();
    }

    const times = [];
    for (let i = 0; i < BENCHMARK_RUNS; i++) {
      const start = performance.now();
      const c = tf.matMul(a, b);
      await c.data(); // Sync
      c.dispose();
      times.push(performance.now() - start);
    }

    a.dispose();
    b.dispose();

    const stats = calculateStats(times);
    results[`matmul_${size}x${size}`] = stats;
    console.log(`   Matmul ${size}x${size}: ${stats.mean.toFixed(2)}ms (p95: ${stats.p95.toFixed(2)}ms)`);
  }

  // Conv2D benchmark
  for (const size of [112, 224]) {
    const input = tf.randomNormal([1, size, size, 3]);
    const kernel = tf.randomNormal([3, 3, 3, 64]);

    // Warmup
    for (let i = 0; i < WARMUP_RUNS; i++) {
      tf.conv2d(input, kernel, 1, 'same').dispose();
    }

    const times = [];
    for (let i = 0; i < BENCHMARK_RUNS; i++) {
      const start = performance.now();
      const out = tf.conv2d(input, kernel, 1, 'same');
      await out.data();
      out.dispose();
      times.push(performance.now() - start);
    }

    input.dispose();
    kernel.dispose();

    const stats = calculateStats(times);
    results[`conv2d_${size}x${size}`] = stats;
    console.log(`   Conv2D ${size}x${size}x3→64: ${stats.mean.toFixed(2)}ms (p95: ${stats.p95.toFixed(2)}ms)`);
  }

  return results;
}

async function benchmarkModelInference(modelName, config) {
  const results = {
    singleInference: null,
    batchInference: {},
    throughput: null
  };

  // Benchmark single inference
  const input = tf.randomNormal(config.inputShape);

  // Simular forward pass com operações típicas de CNN
  async function simulateForward(batchInput) {
    return tf.tidy(() => {
      let x = batchInput;

      // Simular algumas camadas convolucionais
      const k1 = tf.randomNormal([3, 3, 3, 32]);
      x = tf.conv2d(x, k1, 2, 'same');
      x = tf.relu(x);

      const k2 = tf.randomNormal([3, 3, 32, 64]);
      x = tf.conv2d(x, k2, 2, 'same');
      x = tf.relu(x);

      // Global average pooling
      x = tf.mean(x, [1, 2]);

      return x;
    });
  }

  // Warmup
  for (let i = 0; i < WARMUP_RUNS; i++) {
    const out = await simulateForward(input);
    out.dispose();
  }

  // Single inference benchmark
  const singleTimes = [];
  for (let i = 0; i < BENCHMARK_RUNS; i++) {
    const start = performance.now();
    const out = await simulateForward(input);
    await out.data();
    out.dispose();
    singleTimes.push(performance.now() - start);
  }

  results.singleInference = calculateStats(singleTimes);
  input.dispose();

  // Batch inference benchmark
  for (const batchSize of BATCH_SIZES) {
    const batchShape = [batchSize, ...config.inputShape.slice(1)];
    const batchInput = tf.randomNormal(batchShape);

    const batchTimes = [];
    for (let i = 0; i < BENCHMARK_RUNS; i++) {
      const start = performance.now();
      const out = await simulateForward(batchInput);
      await out.data();
      out.dispose();
      batchTimes.push(performance.now() - start);
    }

    results.batchInference[`batch_${batchSize}`] = calculateStats(batchTimes);
    batchInput.dispose();
  }

  // Throughput (images per second)
  const batch8Time = results.batchInference['batch_8']?.mean || results.singleInference.mean * 8;
  results.throughput = (8 / batch8Time) * 1000; // images/sec

  return results;
}

async function benchmarkMemory() {
  console.log('\n💾 Benchmark de Memória\n');

  const results = {};

  // Baseline
  const baseline = tf.memory();
  console.log(`   Baseline: ${(baseline.numBytes / 1024 / 1024).toFixed(2)} MB (${baseline.numTensors} tensors)`);

  // Criar tensores de diferentes tamanhos
  for (const size of [224, 416, 512]) {
    const tensors = [];

    // Simular batch de imagens
    for (let i = 0; i < 8; i++) {
      tensors.push(tf.randomNormal([1, size, size, 3]));
    }

    const mem = tf.memory();
    results[`batch_8_${size}x${size}`] = {
      bytes: mem.numBytes - baseline.numBytes,
      mb: (mem.numBytes - baseline.numBytes) / 1024 / 1024,
      tensors: mem.numTensors - baseline.numTensors
    };

    console.log(`   8x ${size}x${size}x3: ${results[`batch_8_${size}x${size}`].mb.toFixed(2)} MB`);

    // Cleanup
    tensors.forEach(t => t.dispose());
  }

  return results;
}

async function main() {
  console.log('='.repeat(60));
  console.log('Face-API Benchmark Completo');
  console.log('='.repeat(60));
  console.log(`\nBackend: ${tf.getBackend()}`);
  console.log(`TensorFlow.js: ${tf.version.tfjs}`);
  console.log(`Timestamp: ${new Date().toISOString()}`);

  const report = {
    timestamp: new Date().toISOString(),
    backend: tf.getBackend(),
    tfVersion: tf.version.tfjs,
    platform: process.platform,
    nodeVersion: process.version,
    models: {},
    tensorOps: {},
    memory: {},
    summary: {}
  };

  // 1. Carregar info dos modelos
  console.log('\n📦 Modelos Disponíveis\n');

  for (const [modelName, config] of Object.entries(MODEL_CONFIGS)) {
    const modelInfo = await loadModelWeights(modelName);

    if (modelInfo) {
      report.models[modelName] = {
        params: modelInfo.params,
        memoryMB: modelInfo.memoryMB,
        inputShape: config.inputShape,
        description: config.description
      };

      console.log(`   ✓ ${modelName}`);
      console.log(`     ${config.description}`);
      console.log(`     ${modelInfo.params.toLocaleString()} params, ${modelInfo.memoryMB.toFixed(2)} MB`);
      console.log(`     Input: ${config.inputShape.join('x')}`);
    } else {
      console.log(`   ✗ ${modelName} (não encontrado)`);
    }
  }

  // 2. Benchmark operações básicas
  report.tensorOps = await benchmarkTensorOps();

  // 3. Benchmark memória
  report.memory = await benchmarkMemory();

  // 4. Benchmark por modelo (simulado)
  console.log('\n⏱️  Benchmark de Inferência (Simulado)\n');

  for (const [modelName, config] of Object.entries(MODEL_CONFIGS)) {
    if (!report.models[modelName]) continue;

    console.log(`   Benchmarking ${modelName}...`);

    try {
      const results = await benchmarkModelInference(modelName, config);
      report.models[modelName].benchmark = results;

      console.log(`     Single: ${results.singleInference.mean.toFixed(2)}ms (p95: ${results.singleInference.p95.toFixed(2)}ms)`);
      console.log(`     Batch 8: ${results.batchInference['batch_8']?.mean.toFixed(2)}ms`);
      console.log(`     Throughput: ${results.throughput.toFixed(1)} img/s`);
    } catch (e) {
      console.log(`     ⚠️ Erro: ${e.message}`);
    }
  }

  // 5. Resumo
  console.log('\n📈 Resumo\n');

  const summaryTable = [];
  for (const [modelName, modelData] of Object.entries(report.models)) {
    if (modelData.benchmark) {
      summaryTable.push({
        model: modelName,
        params: `${(modelData.params / 1000000).toFixed(2)}M`,
        memory: `${modelData.memoryMB.toFixed(1)}MB`,
        latency: `${modelData.benchmark.singleInference.mean.toFixed(1)}ms`,
        throughput: `${modelData.benchmark.throughput.toFixed(0)}/s`
      });
    }
  }

  // Ordenar por latência
  summaryTable.sort((a, b) => parseFloat(a.latency) - parseFloat(b.latency));

  console.log('   ' + '-'.repeat(70));
  console.log('   | Modelo                    | Params | Memory | Latência | Throughput |');
  console.log('   ' + '-'.repeat(70));
  for (const row of summaryTable) {
    console.log(`   | ${row.model.padEnd(25)} | ${row.params.padStart(6)} | ${row.memory.padStart(6)} | ${row.latency.padStart(8)} | ${row.throughput.padStart(10)} |`);
  }
  console.log('   ' + '-'.repeat(70));

  report.summary = summaryTable;

  // 6. Exportar relatório
  if (!fs.existsSync(OUTPUT_PATH)) {
    fs.mkdirSync(OUTPUT_PATH, { recursive: true });
  }

  const reportPath = path.join(OUTPUT_PATH, `benchmark-${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  console.log(`\n💾 Relatório salvo: ${reportPath}`);

  console.log('\n' + '='.repeat(60));
  console.log('✅ Benchmark concluído!');
  console.log('='.repeat(60));

  // Cleanup
  tf.disposeVariables();
}

main().catch(console.error);

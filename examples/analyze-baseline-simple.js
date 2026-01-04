/**
 * Script simplificado para analisar modelos face-api.
 *
 * Executar no Windows:
 *   node examples/analyze-baseline-simple.js
 */

const tf = require('@tensorflow/tfjs');
const path = require('path');
const fs = require('fs');

// Caminho dos modelos
const MODELS_PATH = path.join(__dirname, '../weights');

async function main() {
  console.log('='.repeat(60));
  console.log('Face-API Model Analyzer (Simple)');
  console.log('='.repeat(60));
  console.log('');

  // Verificar se pasta weights existe
  if (!fs.existsSync(MODELS_PATH)) {
    console.log('❌ Pasta weights/ não encontrada!');
    console.log('   Crie a pasta e coloque os modelos:');
    console.log('   mkdir weights');
    console.log('   Baixe de: https://github.com/vladmandic/face-api/tree/master/model');
    return;
  }

  // Listar arquivos de modelo disponíveis
  console.log('📁 Arquivos em weights/:');
  const files = fs.readdirSync(MODELS_PATH);
  const manifests = files.filter(f => f.endsWith('_manifest.json'));

  if (manifests.length === 0) {
    console.log('   Nenhum modelo encontrado!');
    return;
  }

  manifests.forEach(f => console.log('   ' + f));
  console.log('');

  // Carregar e analisar cada modelo
  console.log('🔍 Analisando modelos...\n');

  for (const manifest of manifests) {
    const modelName = manifest.replace('-weights_manifest.json', '');
    const manifestPath = path.join(MODELS_PATH, manifest);

    try {
      // Ler manifest
      const manifestData = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

      // Calcular total de parâmetros
      let totalParams = 0;
      let totalBytes = 0;
      const layers = [];

      for (const group of manifestData) {
        for (const weight of group.weights) {
          const shape = weight.shape;
          const params = shape.reduce((a, b) => a * b, 1);
          const bytes = params * 4; // float32 = 4 bytes

          totalParams += params;
          totalBytes += bytes;

          layers.push({
            name: weight.name,
            shape: shape,
            params: params
          });
        }
      }

      console.log(`📊 ${modelName}:`);
      console.log(`   Parâmetros: ${totalParams.toLocaleString()}`);
      console.log(`   Memória: ${(totalBytes / 1024 / 1024).toFixed(2)} MB`);
      console.log(`   Camadas: ${layers.length}`);

      // Top 5 maiores camadas
      layers.sort((a, b) => b.params - a.params);
      console.log(`   Maiores camadas:`);
      for (const layer of layers.slice(0, 3)) {
        console.log(`     - ${layer.name}: ${layer.params.toLocaleString()} params [${layer.shape.join('x')}]`);
      }
      console.log('');

    } catch (e) {
      console.log(`   ⚠️ Erro ao analisar ${modelName}: ${e.message}`);
    }
  }

  // Benchmark TensorFlow.js
  console.log('⏱️  Benchmark TensorFlow.js...\n');
  console.log(`   Backend: ${tf.getBackend() || 'cpu'}`);

  // Warmup
  const warmup = tf.randomNormal([100, 100]);
  tf.matMul(warmup, warmup).dispose();
  warmup.dispose();

  // Benchmark
  const size = 500;
  const a = tf.randomNormal([size, size]);
  const b = tf.randomNormal([size, size]);

  const iterations = 20;
  const times = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    const c = tf.matMul(a, b);
    await c.data(); // Force sync
    c.dispose();
    times.push(performance.now() - start);
  }

  a.dispose();
  b.dispose();

  times.sort((x, y) => x - y);
  const mean = times.reduce((x, y) => x + y, 0) / times.length;
  const p50 = times[Math.floor(times.length * 0.5)];
  const p95 = times[Math.floor(times.length * 0.95)];

  console.log(`   Matmul ${size}x${size} (${iterations} runs):`);
  console.log(`     Média: ${mean.toFixed(2)} ms`);
  console.log(`     P50: ${p50.toFixed(2)} ms`);
  console.log(`     P95: ${p95.toFixed(2)} ms`);

  console.log('');
  console.log('='.repeat(60));
  console.log('✅ Análise concluída!');
  console.log('='.repeat(60));
}

main().catch(console.error);

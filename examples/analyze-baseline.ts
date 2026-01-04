/**
 * Script para analisar modelos e criar baseline.
 *
 * Executar:
 *   npx ts-node examples/analyze-baseline.ts
 *
 * Ou com Node.js:
 *   npx tsc examples/analyze-baseline.ts --outDir dist
 *   node dist/analyze-baseline.js
 */

import * as tf from '@tensorflow/tfjs-node'; // Use tfjs-node para melhor performance
import * as faceapi from '../src/index';
import { ModelAnalyzer, QuickAnalysis } from '../src/training';
import * as path from 'path';
import * as fs from 'fs';

// Caminho dos modelos (ajuste conforme necessário)
const MODELS_PATH = path.join(__dirname, '../weights');
const OUTPUT_PATH = path.join(__dirname, '../baselines');

async function main() {
  console.log('='.repeat(60));
  console.log('Face-API Model Analyzer');
  console.log('='.repeat(60));
  console.log();

  // Criar pasta de output se não existir
  if (!fs.existsSync(OUTPUT_PATH)) {
    fs.mkdirSync(OUTPUT_PATH, { recursive: true });
  }

  // ========================================
  // PASSO 1: Carregar os modelos
  // ========================================
  console.log('📦 Passo 1: Carregando modelos...\n');

  try {
    await faceapi.nets.ageGenderNet.loadFromDisk(MODELS_PATH);
    console.log('  ✓ AgeGenderNet carregado');
  } catch (e) {
    console.log('  ✗ AgeGenderNet não encontrado em', MODELS_PATH);
  }

  try {
    await faceapi.nets.faceExpressionNet.loadFromDisk(MODELS_PATH);
    console.log('  ✓ FaceExpressionNet carregado');
  } catch (e) {
    console.log('  ✗ FaceExpressionNet não encontrado');
  }

  try {
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODELS_PATH);
    console.log('  ✓ FaceRecognitionNet carregado');
  } catch (e) {
    console.log('  ✗ FaceRecognitionNet não encontrado');
  }

  try {
    await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_PATH);
    console.log('  ✓ FaceLandmark68Net carregado');
  } catch (e) {
    console.log('  ✗ FaceLandmark68Net não encontrado');
  }

  try {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODELS_PATH);
    console.log('  ✓ SsdMobilenetv1 carregado');
  } catch (e) {
    console.log('  ✗ SsdMobilenetv1 não encontrado');
  }

  console.log();

  // ========================================
  // PASSO 2: Analisar arquitetura
  // ========================================
  console.log('🔍 Passo 2: Analisando arquitetura dos modelos...\n');

  const models = [
    { name: 'AgeGenderNet', net: faceapi.nets.ageGenderNet },
    { name: 'FaceExpressionNet', net: faceapi.nets.faceExpressionNet },
    { name: 'FaceRecognitionNet', net: faceapi.nets.faceRecognitionNet },
    { name: 'FaceLandmark68Net', net: faceapi.nets.faceLandmark68Net },
    { name: 'SsdMobilenetv1', net: faceapi.nets.ssdMobilenetv1 },
  ];

  const analyses: Record<string, any> = {};

  for (const { name, net } of models) {
    if (!net.isLoaded) {
      console.log(`  ⏭ ${name}: não carregado, pulando...`);
      continue;
    }

    try {
      const analysis = await ModelAnalyzer.analyze(net);
      analyses[name] = analysis;

      console.log(`  📊 ${name}:`);
      console.log(`     Parâmetros: ${analysis.totalParams.toLocaleString()}`);
      console.log(`     Treináveis: ${analysis.trainableParams.toLocaleString()}`);
      console.log(`     Memória: ${analysis.memoryMB.toFixed(2)} MB`);
      console.log();
    } catch (e) {
      console.log(`  ✗ Erro ao analisar ${name}:`, e);
    }
  }

  // ========================================
  // PASSO 3: Health Check
  // ========================================
  console.log('🩺 Passo 3: Verificando saúde dos pesos...\n');

  for (const { name, net } of models) {
    if (!net.isLoaded) continue;

    try {
      const health = await QuickAnalysis.healthCheck(net);

      if (health.healthy) {
        console.log(`  ✓ ${name}: Saudável`);
      } else {
        console.log(`  ⚠ ${name}: ${health.issues.length} problemas encontrados`);
        for (const issue of health.issues.slice(0, 3)) {
          console.log(`    - ${issue}`);
        }
        if (health.issues.length > 3) {
          console.log(`    ... e mais ${health.issues.length - 3} problemas`);
        }
      }
    } catch (e) {
      console.log(`  ✗ Erro no health check de ${name}`);
    }
  }

  console.log();

  // ========================================
  // PASSO 4: Benchmark de Latência
  // ========================================
  console.log('⏱️  Passo 4: Benchmark de latência...\n');

  // Benchmark AgeGenderNet
  if (faceapi.nets.ageGenderNet.isLoaded) {
    console.log('  Benchmarking AgeGenderNet...');

    const latency = await ModelAnalyzer.benchmarkLatency(
      faceapi.nets.ageGenderNet,
      () => tf.randomNormal([1, 112, 112, 3]), // Input shape para AgeGenderNet
      (input) => {
        const result = faceapi.nets.ageGenderNet.forwardInput(
          new faceapi.NetInput([input as tf.Tensor3D])
        );
        // Retornar tensor combinado para dispose
        return tf.concat([result.age.expandDims(), result.gender], 1);
      },
      { warmupRuns: 3, benchmarkRuns: 20 }
    );

    console.log(`  📊 AgeGenderNet Latência:`);
    console.log(`     Média: ${latency.mean.toFixed(2)} ms`);
    console.log(`     Std: ${latency.std.toFixed(2)} ms`);
    console.log(`     P50: ${latency.p50.toFixed(2)} ms`);
    console.log(`     P95: ${latency.p95.toFixed(2)} ms`);
    console.log();
  }

  // Benchmark FaceExpressionNet
  if (faceapi.nets.faceExpressionNet.isLoaded) {
    console.log('  Benchmarking FaceExpressionNet...');

    const latency = await ModelAnalyzer.benchmarkLatency(
      faceapi.nets.faceExpressionNet,
      () => tf.randomNormal([1, 112, 112, 3]),
      (input) => faceapi.nets.faceExpressionNet.forwardInput(input as tf.Tensor4D),
      { warmupRuns: 3, benchmarkRuns: 20 }
    );

    console.log(`  📊 FaceExpressionNet Latência:`);
    console.log(`     Média: ${latency.mean.toFixed(2)} ms`);
    console.log(`     Std: ${latency.std.toFixed(2)} ms`);
    console.log(`     P50: ${latency.p50.toFixed(2)} ms`);
    console.log(`     P95: ${latency.p95.toFixed(2)} ms`);
    console.log();
  }

  // ========================================
  // PASSO 5: Exportar relatório
  // ========================================
  console.log('💾 Passo 5: Exportando relatórios...\n');

  const report = {
    timestamp: new Date().toISOString(),
    tfVersion: tf.version.tfjs,
    models: analyses
  };

  const reportPath = path.join(OUTPUT_PATH, `baseline-${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`  ✓ Relatório salvo em: ${reportPath}`);

  // Salvar relatório legível
  let textReport = `Face-API Baseline Report\n`;
  textReport += `Generated: ${report.timestamp}\n`;
  textReport += `TensorFlow.js: ${report.tfVersion}\n\n`;

  for (const [name, analysis] of Object.entries(analyses)) {
    textReport += (analysis as any).summary + '\n\n';
  }

  const textPath = path.join(OUTPUT_PATH, `baseline-${Date.now()}.txt`);
  fs.writeFileSync(textPath, textReport);
  console.log(`  ✓ Relatório texto salvo em: ${textPath}`);

  console.log();
  console.log('='.repeat(60));
  console.log('✅ Análise concluída!');
  console.log('='.repeat(60));

  // Cleanup
  tf.disposeVariables();
}

// Executar
main().catch(console.error);

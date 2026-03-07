# Guia de Uso: Reconhecimento Facial para Ações

## Instalação Rápida

```bash
# Instalar a biblioteca
npm install @anthropic/face-api

# OU usar diretamente do seu projeto local
npm link /caminho/para/face-api
```

## Modelos Necessários

Baixe os modelos e coloque em uma pasta `/models` ou `/public/models`:

| Modelo | Tamanho | Função |
|--------|---------|--------|
| `ssd_mobilenetv1_model-weights_manifest.json` | ~5.4MB | Detecção de faces |
| `face_landmark_68_model-weights_manifest.json` | ~350KB | Pontos faciais |
| `face_recognition_model-weights_manifest.json` | ~6.2MB | Embeddings/Descritores |

**Alternativas de Detecção:**
- `tiny_face_detector_model` (~190KB) - Mais rápido, menos preciso
- `blazeface_model` (~400KB) - Ultra-rápido, bom para mobile

## Uso Básico (Mínimo)

```typescript
import * as faceapi from '@anthropic/face-api';

// 1. Carregar modelos
await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
await faceapi.nets.faceRecognitionNet.loadFromUri('/models');

// 2. Cadastrar pessoa (extrair descritor de uma foto)
const imgPessoa = document.getElementById('foto') as HTMLImageElement;
const detection = await faceapi
  .detectSingleFace(imgPessoa)
  .withFaceLandmarks()
  .withFaceDescriptor();

if (detection) {
  const descritor = detection.descriptor; // Float32Array[128]
  // Salve este descritor no banco de dados
  localStorage.setItem('joao_descriptor', JSON.stringify(Array.from(descritor)));
}

// 3. Reconhecer pessoa
const video = document.getElementById('webcam') as HTMLVideoElement;
const detectionVideo = await faceapi
  .detectSingleFace(video)
  .withFaceLandmarks()
  .withFaceDescriptor();

if (detectionVideo) {
  // Carregar descritores salvos
  const joaoDescriptor = new Float32Array(
    JSON.parse(localStorage.getItem('joao_descriptor')!)
  );

  // Calcular distância (menor = mais similar)
  const distance = faceapi.euclideanDistance(
    detectionVideo.descriptor,
    joaoDescriptor
  );

  if (distance < 0.5) {
    console.log('João reconhecido! Abrindo portão...');
    // Executar ação
  }
}
```

## Uso com FaceMatcher (Múltiplas Pessoas)

```typescript
import * as faceapi from '@anthropic/face-api';

// Criar matcher com pessoas cadastradas
const labeledDescriptors = [
  new faceapi.LabeledFaceDescriptors('João', [joaoDescriptor]),
  new faceapi.LabeledFaceDescriptors('Maria', [mariaDescriptor]),
  new faceapi.LabeledFaceDescriptors('Pedro', [pedroDescriptor]),
];

const matcher = new faceapi.FaceMatcher(labeledDescriptors, 0.5);

// Reconhecer
const detection = await faceapi
  .detectSingleFace(video)
  .withFaceLandmarks()
  .withFaceDescriptor();

if (detection) {
  const match = matcher.findBestMatch(detection.descriptor);

  if (match.label !== 'unknown') {
    console.log(`Reconhecido: ${match.label} (${((1 - match.distance) * 100).toFixed(1)}%)`);
    // Executar ação baseada na pessoa
  }
}
```

## Exportações Principais

```typescript
// Detecção de faces
faceapi.detectSingleFace(input, options)
faceapi.detectAllFaces(input, options)

// Opções de detecção
new faceapi.SsdMobilenetv1Options({ minConfidence: 0.7 })
new faceapi.TinyFaceDetectorOptions({ inputSize: 416 })

// Chaining
.withFaceLandmarks()
.withFaceDescriptor()
.withFaceExpressions()
.withAgeAndGender()

// Matching
new faceapi.FaceMatcher(labeledDescriptors, threshold)
new faceapi.LabeledFaceDescriptors(label, descriptors)

// Utilitários
faceapi.euclideanDistance(descriptor1, descriptor2)
faceapi.resizeResults(results, { width, height })

// Redes
faceapi.nets.ssdMobilenetv1
faceapi.nets.tinyFaceDetector
faceapi.nets.faceLandmark68Net
faceapi.nets.faceRecognitionNet
faceapi.nets.faceExpressionNet
faceapi.nets.ageGenderNet
```

## Integração com Hardware (Exemplos)

### Node.js + GPIO (Raspberry Pi)

```typescript
import * as faceapi from '@anthropic/face-api';
import { Gpio } from 'onoff';

const relayPortao = new Gpio(17, 'out');

async function abrirPortao(pessoa: string) {
  console.log(`Abrindo portão para ${pessoa}`);
  relayPortao.writeSync(1);
  await new Promise(r => setTimeout(r, 3000));
  relayPortao.writeSync(0);
}
```

### API REST

```typescript
async function executarAcao(pessoa: string, acao: string) {
  await fetch('/api/acoes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pessoa, acao, timestamp: Date.now() }),
  });
}
```

### WebSocket (Tempo Real)

```typescript
const ws = new WebSocket('ws://localhost:8080');

function notificarReconhecimento(pessoa: string) {
  ws.send(JSON.stringify({
    type: 'face_recognized',
    pessoa,
    timestamp: Date.now(),
  }));
}
```

## Estrutura de Dados Recomendada

```typescript
// Pessoa cadastrada
interface PessoaCadastrada {
  id: string;
  nome: string;
  descriptor: number[]; // Float32Array serializado
  permissoes: string[]; // ['portao', 'alarme', 'luzes']
  ativo: boolean;
  criadoEm: Date;
}

// Log de acesso
interface LogAcesso {
  id: string;
  pessoaId: string;
  pessoaNome: string;
  acao: string;
  confianca: number;
  autorizado: boolean;
  timestamp: Date;
}
```

## Performance Tips

1. **Use TinyFaceDetector para real-time** - 5-10x mais rápido
2. **Reduza resolução do vídeo** - 640x480 é suficiente
3. **Limite taxa de detecção** - 2-5 FPS é suficiente para controle de acesso
4. **Pre-carregue modelos** - Carregue no início da aplicação
5. **Use WebGL backend** - `await faceapi.tf.setBackend('webgl')`

## Segurança

- **Armazene descritores com criptografia** em produção
- **Adicione liveness detection** para evitar fotos/vídeos
- **Log todas as tentativas** de acesso (autorizadas ou não)
- **Defina timeout** para tentativas repetidas
- **Use HTTPS** para transmissão de dados

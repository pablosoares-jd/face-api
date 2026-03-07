# KYC BlazeFace + AdaFace - Problemas e Incompatibilidades

**Data:** 2026-01-14
**Status:** Em investigação - NÃO RESOLVIDO

---

## Problema Principal

```
TypeError: object null is not iterable (cannot read property Symbol(Symbol.iterator))
```

O erro ocorre em chamadas `Array.from()` quando os tensores retornam `null` ou `undefined` durante a detecção/reconhecimento facial.

---

## Contexto

### Objetivo Original
Implementar KYC (Know Your Customer) com:
1. **Step 1:** Captura de selfie com detecção facial em tempo real (FUNCIONANDO)
2. **Step 2:** Verificação de identidade comparando selfie vs documento (FALHANDO)

### Stack Utilizada
- **Detector primário:** BlazeFace (mais rápido, keypoints)
- **Detector fallback:** SSD MobileNetv1 (mais compatível)
- **Reconhecedor primário:** AdaFace (512-dim, mais preciso)
- **Reconhecedor fallback:** FaceNet/face_recognition_model (128-dim)

---

## Fluxo de Fallback

```
BlazeFace (graph-model)
    ↓ 404 - modelo não encontrado
SSD MobileNetv1 (layers-model)
    ↓ Carrega com sucesso

AdaFace (graph-model)
    ↓ 404 - modelo não encontrado
FaceNet (layers-model)
    ↓ Carrega com sucesso
```

O problema é que **mesmo com fallback funcionando**, o erro `Array.from(null)` persiste.

---

## Arquivos Modificados (Tentativas de Correção)

### 1. `src/blazeFace/BlazeFace.ts` (linha ~434)
```typescript
// ANTES:
const scoresData = await scores.data();
const selectedIndices = this.nonMaxSuppression(
  decodedBoxes,
  Array.from(scoresData), // ❌ Pode ser null
  ...
);

// DEPOIS:
const scoresData = await scores.data();
if (!scoresData || scoresData.length === 0) {
  console.warn('BlazeFace: No scores data returned from model');
  return [];
}
const selectedIndices = this.nonMaxSuppression(
  decodedBoxes,
  Array.from(scoresData), // ✅ Protegido
  ...
);
```

### 2. `src/ssdMobilenetv1/SsdMobilenetv1.ts` (linha ~59)
```typescript
// ANTES:
const scoresArray = Array.from(scoresData); // ❌ scoresData pode ser null

// DEPOIS:
if (!scoresData || scoresData.length === 0) {
  console.warn('SsdMobilenetv1: No scores data returned from model');
  return [];
}
if (!boxesData || boxesData.length === 0) {
  console.warn('SsdMobilenetv1: No boxes data returned from model');
  return [];
}
const scoresArray = Array.from(scoresData); // ✅ Protegido
```

### 3. `src/faceRecognitionNet/FaceRecognitionNet.ts` (linha ~69)
```typescript
// ANTES:
return netInput.isBatchInput ? faceDescriptorsForBatch : faceDescriptorsForBatch[0];
// ❌ Se array vazio, faceDescriptorsForBatch[0] = undefined

// DEPOIS:
if (!faceDescriptorTensors || faceDescriptorTensors.length === 0) {
  console.warn('FaceRecognitionNet: No face descriptor tensors produced');
  return new Float32Array(128);
}
// ... validações adicionais ...
if (!netInput.isBatchInput) {
  const descriptor = faceDescriptorsForBatch[0];
  if (!descriptor) {
    console.warn('FaceRecognitionNet: First descriptor is null/undefined');
    return new Float32Array(128);
  }
  return descriptor;
}
```

### 4. `src/adaFace/AdaFace.ts` (linha ~220)
```typescript
// Mesma lógica de proteção que FaceRecognitionNet
if (!descriptorTensors || descriptorTensors.length === 0) {
  console.warn('AdaFace: No descriptor tensors produced');
  return new Float32Array(512);
}
// ... validações ...
```

### 5. `src/euclideanDistance.ts`
```typescript
// ANTES:
export function euclideanDistance(arr1: number[] | Float32Array, arr2: number[] | Float32Array) {
  const desc1 = Array.from(arr1); // ❌ arr1 pode ser null

// DEPOIS:
export function euclideanDistance(
  arr1: number[] | Float32Array | null | undefined,
  arr2: number[] | Float32Array | null | undefined
): number {
  if (!arr1 || !arr2) {
    throw new Error('euclideanDistance: arr1 and arr2 must not be null or undefined');
  }
  // ...
}
```

### 6. `src/kyc/KYCDocumentAnalyzer.ts`
- Adicionado verificação de descriptors antes de chamar euclideanDistance
- Adicionado `detectWithOfficialBlazeFace()` como fallback adicional
- Tratamento de Tensor vs Array nos outputs do BlazeFace oficial

---

## Incompatibilidades Identificadas

### 1. BlazeFace Graph Model vs Layers Model
- O builder do face-api gera **layers-model** por padrão
- O BlazeFace oficial do TensorFlow Hub usa **graph-model**
- Formatos incompatíveis: `blazeface_model-weights_manifest.json` espera graph-model

### 2. AdaFace Model Format
- AdaFace precisa de modelo treinado específico
- Não existe `adaface_model-weights_manifest.json` no builder padrão
- Fallback para FaceNet funciona, mas gera 128-dim em vez de 512-dim

### 3. Tensor.data() Retornando Null
- Em certas condições (modelo não carregado, input inválido), `tensor.data()` retorna null
- O código original não tinha proteção para isso
- Mesmo com proteções, o erro persiste em algum ponto não identificado

---

## O Que Ainda Falta Investigar

1. **Localização exata do erro:** O erro ocorre na linha ~10350 do bundle minificado. Precisamos identificar qual `Array.from` específico está falhando.

2. **Condição de corrida:** Pode haver race condition entre carregamento de modelo e inferência.

3. **Input inválido:** Verificar se o input da imagem está chegando corretamente (canvas, dimensões, etc).

4. **Build pipeline:** O builder pode estar gerando código incorreto ou omitindo validações.

---

## Como Reproduzir

1. Iniciar o projeto studio_jumbodigital
2. Acessar a página de verificação de identidade como customer
3. Fazer upload de selfie e documento
4. Clicar em "Verificar"
5. Erro aparece no console

---

## Solução Recomendada: Usar SSD + FaceNet

**Status:** FUNCIONAL ✅

Os modelos que **funcionam** com o builder atual:
- **Detector:** SSD MobileNetv1 (`ssd_mobilenetv1_model`)
- **Reconhecedor:** FaceNet (`face_recognition_model`)
- **Landmarks:** `face_landmark_68_model`

### Implementação

```typescript
// Em KYCDocumentAnalyzer.ts - forçar uso de SSD + FaceNet
// Não tentar carregar BlazeFace/AdaFace

// Detector
this.detector = new SsdMobilenetv1(); // Não usar BlazeFace
await this.detector.load(modelPath);

// Reconhecedor
this.recognitionNet = new FaceRecognitionNet(); // Não usar AdaFace
await this.recognitionNet.load(modelPath);
```

### Trade-offs

| Aspecto | BlazeFace + AdaFace | SSD + FaceNet |
|---------|---------------------|---------------|
| Velocidade | ~200-1000 FPS | ~20-40 FPS |
| Precisão | 512-dim descriptor | 128-dim descriptor |
| Keypoints | 6 pontos faciais | Não |
| Compatibilidade | ❌ Problemas | ✅ Funciona |
| Status | Experimental | Estável |

**Recomendação:** Usar SSD + FaceNet até que os modelos BlazeFace/AdaFace estejam no formato correto.

---

## Workarounds Alternativos (Futuro)

### Opção A: Gerar modelos compatíveis
- Converter BlazeFace para layers-model
- Treinar/converter AdaFace para formato compatível
- Atualizar manifest files

### Opção B: Usar @tensorflow-models/blazeface diretamente
- Já foi integrado em `KYCDocumentAnalyzer.detectWithOfficialBlazeFace()`
- Funciona para detecção, mas não resolve o problema de reconhecimento

---

## Arquivos de Referência

- `src/kyc/KYCDocumentAnalyzer.ts` - Analyzer principal
- `src/kyc/KYCFaceAnalyzer.ts` - Analyzer alternativo (mais simples)
- `src/blazeFace/BlazeFace.ts` - Detector BlazeFace
- `src/adaFace/AdaFace.ts` - Reconhecedor AdaFace
- `src/faceRecognitionNet/FaceRecognitionNet.ts` - Reconhecedor FaceNet

---

## Conclusão

O problema é uma combinação de:
1. Modelos BlazeFace/AdaFace não disponíveis no formato esperado
2. Fallbacks funcionando mas algo ainda retorna null em algum ponto
3. Falta de validação defensiva em todos os pontos de `Array.from()`

As correções aplicadas adicionaram validações defensivas, mas o erro persiste, indicando que há outro ponto no código onde `Array.from()` é chamado com valor null que ainda não foi identificado.

**Próximos passos:** Usar source maps para identificar a linha exata do erro no código não-minificado.

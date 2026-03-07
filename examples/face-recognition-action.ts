/**
 * Exemplo: Sistema de Reconhecimento Facial para Ações
 * Use case: Abrir portão, executar comandos, controle de acesso
 *
 * Este exemplo mostra como usar o face-api para:
 * 1. Cadastrar rostos autorizados
 * 2. Detectar e reconhecer rostos em tempo real
 * 3. Executar ações baseadas no reconhecimento
 */

import * as faceapi from '../src/index';

// ============================================
// TIPOS E INTERFACES
// ============================================

interface AuthorizedPerson {
  id: string;
  name: string;
  descriptor: Float32Array;
  permissions: string[]; // ex: ['gate', 'door', 'alarm']
}

interface RecognitionResult {
  recognized: boolean;
  person?: AuthorizedPerson;
  confidence: number;
  action?: string;
}

interface ActionConfig {
  action: string;
  requiredPermission: string;
  callback: (person: AuthorizedPerson) => Promise<void>;
}

// ============================================
// CLASSE PRINCIPAL: FaceAccessControl
// ============================================

export class FaceAccessControl {
  private matcher: faceapi.FaceMatcher | null = null;
  private authorizedPeople: Map<string, AuthorizedPerson> = new Map();
  private actions: Map<string, ActionConfig> = new Map();
  private isInitialized = false;

  // Thresholds
  private readonly RECOGNITION_THRESHOLD = 0.5; // Menor = mais restritivo
  private readonly DETECTION_CONFIDENCE = 0.7;

  /**
   * Inicializa os modelos de IA necessários.
   * Chame isso uma vez no início da aplicação.
   */
  async initialize(modelsPath: string): Promise<void> {
    console.log('Carregando modelos de reconhecimento facial...');

    try {
      // Carregar modelos necessários em paralelo
      await Promise.all([
        // Detector de faces (escolha um):
        faceapi.nets.ssdMobilenetv1.loadFromUri(modelsPath),
        // OU para melhor performance:
        // faceapi.nets.tinyFaceDetector.loadFromUri(modelsPath),

        // Detector de landmarks (pontos faciais)
        faceapi.nets.faceLandmark68Net.loadFromUri(modelsPath),

        // Gerador de embeddings/descritores faciais
        faceapi.nets.faceRecognitionNet.loadFromUri(modelsPath),
      ]);

      this.isInitialized = true;
      console.log('Modelos carregados com sucesso!');
    } catch (error) {
      throw new Error(`Falha ao carregar modelos: ${error}`);
    }
  }

  /**
   * Cadastra uma nova pessoa autorizada a partir de uma imagem.
   */
  async registerPerson(
    name: string,
    imageSource: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
    permissions: string[] = ['gate'],
  ): Promise<AuthorizedPerson> {
    this.ensureInitialized();

    // Detectar face e extrair descritor
    const detection = await faceapi
      .detectSingleFace(imageSource, new faceapi.SsdMobilenetv1Options({ minConfidence: this.DETECTION_CONFIDENCE }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      throw new Error(`Nenhuma face detectada na imagem para ${name}`);
    }

    const person: AuthorizedPerson = {
      id: `person_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      descriptor: detection.descriptor,
      permissions,
    };

    this.authorizedPeople.set(person.id, person);
    this.rebuildMatcher();

    console.log(`Pessoa cadastrada: ${name} (${permissions.join(', ')})`);
    return person;
  }

  /**
   * Cadastra pessoa com múltiplas fotos (mais preciso).
   */
  async registerPersonMultiplePhotos(
    name: string,
    images: Array<HTMLImageElement | HTMLCanvasElement>,
    permissions: string[] = ['gate'],
  ): Promise<AuthorizedPerson> {
    this.ensureInitialized();

    const descriptors: Float32Array[] = [];

    for (const image of images) {
      const detection = await faceapi
        .detectSingleFace(image, new faceapi.SsdMobilenetv1Options({ minConfidence: this.DETECTION_CONFIDENCE }))
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        descriptors.push(detection.descriptor);
      }
    }

    if (descriptors.length === 0) {
      throw new Error(`Nenhuma face detectada nas imagens para ${name}`);
    }

    // Calcular descritor médio para maior robustez
    const avgDescriptor = this.computeAverageDescriptor(descriptors);

    const person: AuthorizedPerson = {
      id: `person_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      descriptor: avgDescriptor,
      permissions,
    };

    this.authorizedPeople.set(person.id, person);
    this.rebuildMatcher();

    console.log(`Pessoa cadastrada com ${descriptors.length} fotos: ${name}`);
    return person;
  }

  /**
   * Tenta reconhecer uma pessoa em uma imagem/vídeo.
   */
  async recognize(
    imageSource: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  ): Promise<RecognitionResult> {
    this.ensureInitialized();

    if (!this.matcher) {
      return { recognized: false, confidence: 0 };
    }

    // Detectar face
    const detection = await faceapi
      .detectSingleFace(imageSource, new faceapi.SsdMobilenetv1Options({ minConfidence: this.DETECTION_CONFIDENCE }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      return { recognized: false, confidence: 0 };
    }

    // Encontrar melhor match
    const match = this.matcher.findBestMatch(detection.descriptor);
    const confidence = 1 - match.distance; // Converter distância para confiança

    if (match.label === 'unknown') {
      return { recognized: false, confidence };
    }

    // Encontrar pessoa pelo nome
    const person = Array.from(this.authorizedPeople.values())
      .find(p => p.name === match.label);

    return {
      recognized: true,
      person,
      confidence,
    };
  }

  /**
   * Reconhece e executa ação se autorizado.
   */
  async recognizeAndAct(
    imageSource: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
    actionName: string,
  ): Promise<RecognitionResult> {
    const result = await this.recognize(imageSource);

    if (!result.recognized || !result.person) {
      console.log('Pessoa não reconhecida. Acesso negado.');
      return result;
    }

    const action = this.actions.get(actionName);
    if (!action) {
      console.log(`Ação '${actionName}' não configurada.`);
      return result;
    }

    // Verificar permissão
    if (!result.person.permissions.includes(action.requiredPermission)) {
      console.log(`${result.person.name} não tem permissão para '${actionName}'.`);
      return { ...result, action: 'denied' };
    }

    // Executar ação
    console.log(`Executando '${actionName}' para ${result.person.name}...`);
    await action.callback(result.person);

    return { ...result, action: actionName };
  }

  /**
   * Registra uma ação que pode ser executada após reconhecimento.
   */
  registerAction(
    actionName: string,
    requiredPermission: string,
    callback: (person: AuthorizedPerson) => Promise<void>,
  ): void {
    this.actions.set(actionName, {
      action: actionName,
      requiredPermission,
      callback,
    });
  }

  /**
   * Exporta dados para persistência (salvar em arquivo/banco).
   */
  exportData(): string {
    const data = Array.from(this.authorizedPeople.values()).map(p => ({
      id: p.id,
      name: p.name,
      descriptor: Array.from(p.descriptor),
      permissions: p.permissions,
    }));
    return JSON.stringify(data);
  }

  /**
   * Importa dados previamente salvos.
   */
  importData(jsonData: string): void {
    const data = JSON.parse(jsonData) as Array<{
      id: string;
      name: string;
      descriptor: number[];
      permissions: string[];
    }>;

    this.authorizedPeople.clear();

    for (const item of data) {
      const person: AuthorizedPerson = {
        id: item.id,
        name: item.name,
        descriptor: new Float32Array(item.descriptor),
        permissions: item.permissions,
      };
      this.authorizedPeople.set(person.id, person);
    }

    this.rebuildMatcher();
    console.log(`Importados ${data.length} registros.`);
  }

  // ============================================
  // MÉTODOS PRIVADOS
  // ============================================

  private ensureInitialized(): void {
    if (!this.isInitialized) {
      throw new Error('FaceAccessControl não inicializado. Chame initialize() primeiro.');
    }
  }

  private rebuildMatcher(): void {
    const people = Array.from(this.authorizedPeople.values());

    if (people.length === 0) {
      this.matcher = null;
      return;
    }

    const labeledDescriptors = people.map(
      p => new faceapi.LabeledFaceDescriptors(p.name, [p.descriptor]),
    );

    this.matcher = new faceapi.FaceMatcher(labeledDescriptors, this.RECOGNITION_THRESHOLD);
  }

  private computeAverageDescriptor(descriptors: Float32Array[]): Float32Array {
    const length = descriptors[0].length;
    const avg = new Float32Array(length);

    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (const desc of descriptors) {
        sum += desc[i];
      }
      avg[i] = sum / descriptors.length;
    }

    return avg;
  }
}

// ============================================
// EXEMPLO DE USO: CONTROLE DE PORTÃO
// ============================================

async function exemploControlePortao() {
  const accessControl = new FaceAccessControl();

  // 1. Inicializar modelos (faça isso no início da aplicação)
  await accessControl.initialize('/models');

  // 2. Registrar ações disponíveis
  accessControl.registerAction('abrir_portao', 'gate', async (person) => {
    console.log(`🚪 Abrindo portão para ${person.name}...`);
    // Aqui você chamaria sua API/hardware
    // await fetch('/api/gate/open', { method: 'POST' });
    // ou GPIO em Raspberry Pi, etc.
  });

  accessControl.registerAction('desarmar_alarme', 'alarm', async (person) => {
    console.log(`🔓 Desarmando alarme para ${person.name}...`);
    // await fetch('/api/alarm/disarm', { method: 'POST' });
  });

  // 3. Cadastrar pessoas autorizadas (faça isso uma vez)
  // const imgJoao = document.getElementById('foto-joao') as HTMLImageElement;
  // await accessControl.registerPerson('João', imgJoao, ['gate', 'alarm']);

  // 4. Em tempo real (loop de vídeo):
  // const video = document.getElementById('webcam') as HTMLVideoElement;
  // const result = await accessControl.recognizeAndAct(video, 'abrir_portao');
  // if (result.recognized) {
  //   console.log(`Bem-vindo, ${result.person?.name}! (${(result.confidence * 100).toFixed(1)}%)`);
  // }

  // 5. Persistir dados (salve em localStorage, arquivo, banco...)
  // const dados = accessControl.exportData();
  // localStorage.setItem('face-access-data', dados);

  // 6. Restaurar dados salvos
  // const dadosSalvos = localStorage.getItem('face-access-data');
  // if (dadosSalvos) accessControl.importData(dadosSalvos);
}

// ============================================
// EXEMPLO DE USO: LOOP DE VÍDEO EM TEMPO REAL
// ============================================

async function exemploVideoLoop() {
  const accessControl = new FaceAccessControl();
  await accessControl.initialize('/models');

  // Configurar ações
  accessControl.registerAction('abrir_portao', 'gate', async (person) => {
    console.log(`Portão aberto para: ${person.name}`);
  });

  // Obter vídeo da webcam
  const video = document.createElement('video');
  video.autoplay = true;
  video.muted = true;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: 640, height: 480 },
  });
  video.srcObject = stream;

  // Aguardar vídeo carregar
  await new Promise<void>(resolve => {
    video.onloadedmetadata = () => resolve();
  });

  // Loop de reconhecimento
  let lastRecognitionTime = 0;
  const RECOGNITION_INTERVAL = 2000; // 2 segundos entre tentativas

  const recognitionLoop = async () => {
    const now = Date.now();

    if (now - lastRecognitionTime >= RECOGNITION_INTERVAL) {
      lastRecognitionTime = now;

      try {
        const result = await accessControl.recognizeAndAct(video, 'abrir_portao');

        if (result.recognized) {
          console.log(`✅ ${result.person?.name} - Confiança: ${(result.confidence * 100).toFixed(1)}%`);
        } else {
          console.log('❌ Pessoa não reconhecida');
        }
      } catch (error) {
        console.error('Erro no reconhecimento:', error);
      }
    }

    requestAnimationFrame(recognitionLoop);
  };

  recognitionLoop();
}

// Export para uso em outros projetos
export {
  type AuthorizedPerson,
  type RecognitionResult,
  type ActionConfig,
};

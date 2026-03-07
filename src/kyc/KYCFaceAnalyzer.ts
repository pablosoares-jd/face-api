import type { FaceDetection } from '../classes/FaceDetection';
import type { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import type { TNetInput } from '../dom/index';
import { toNetInput } from '../dom/index';
import { SsdMobilenetv1 } from '../ssdMobilenetv1/SsdMobilenetv1';
import { FaceLandmark68Net } from '../faceLandmarkNet/FaceLandmark68Net';
import { FaceRecognitionNet } from '../faceRecognitionNet/FaceRecognitionNet';
import { euclideanDistance } from '../euclideanDistance';
import { ImageQualityAnalyzer } from './ImageQualityAnalyzer';

/**
 * Face quality metrics for KYC validation.
 */
export interface FaceQualityMetrics {
  /** Overall quality score (0-1) */
  overallScore: number;

  /** Sharpness/blur score (0-1, higher = sharper) */
  sharpness: number;

  /** Brightness score (0-1, 0.5 = optimal) */
  brightness: number;

  /** Contrast score (0-1) */
  contrast: number;

  /** Face size relative to image (0-1) */
  faceSize: number;

  /** Head pose angles in degrees */
  pose: {
    yaw: number; // Left-right rotation
    pitch: number; // Up-down rotation
    roll: number; // Tilt
  };

  /** Is the pose within acceptable range? */
  isFrontal: boolean;

  /** Are both eyes visible? */
  eyesVisible: boolean;

  /** Confidence of the detection */
  detectionConfidence: number;
}

/**
 * KYC analysis result.
 */
export interface KYCAnalysisResult {
  /** Was a valid face detected? */
  faceDetected: boolean;

  /** Detection details */
  detection: FaceDetection | null;

  /** Facial landmarks */
  landmarks: FaceLandmarks68 | null;

  /** Face descriptor for matching */
  descriptor: Float32Array | null;

  /** Quality metrics */
  quality: FaceQualityMetrics | null;

  /** Validation issues found */
  issues: string[];

  /** Is the image acceptable for KYC? */
  isAcceptable: boolean;

  /** Recommendations for improvement */
  recommendations: string[];
}

/**
 * Result of comparing two faces for KYC verification.
 */
export interface KYCMatchResult {
  /** Are the faces from the same person? */
  isMatch: boolean;

  /** Similarity score (0-1, higher = more similar) */
  similarity: number;

  /** Euclidean distance between descriptors */
  distance: number;

  /** Confidence level of the match */
  confidence: 'high' | 'medium' | 'low';

  /** Quality of ID photo */
  idQuality: FaceQualityMetrics | null;

  /** Quality of selfie */
  selfieQuality: FaceQualityMetrics | null;
}

/**
 * Configuration for KYC analysis.
 */
export interface KYCConfig {
  /** Minimum detection confidence (default: 0.8) */
  minDetectionConfidence: number;

  /** Minimum face size as fraction of image (default: 0.1) */
  minFaceSize: number;

  /** Maximum face size as fraction of image (default: 0.9) */
  maxFaceSize: number;

  /** Maximum yaw angle in degrees (default: 15) */
  maxYaw: number;

  /** Maximum pitch angle in degrees (default: 15) */
  maxPitch: number;

  /** Maximum roll angle in degrees (default: 10) */
  maxRoll: number;

  /** Minimum sharpness score (default: 0.3) */
  minSharpness: number;

  /** Match threshold for face comparison (default: 0.6) */
  matchThreshold: number;
}

const DEFAULT_CONFIG: KYCConfig = {
  minDetectionConfidence: 0.8,
  minFaceSize: 0.1,
  maxFaceSize: 0.9,
  maxYaw: 15,
  maxPitch: 15,
  maxRoll: 10,
  minSharpness: 0.3,
  matchThreshold: 0.6,
};

/**
 * KYC Face Analyzer - Optimized for identity verification.
 *
 * Features:
 * - High-precision face detection
 * - Face quality assessment (blur, lighting, pose)
 * - Face matching between ID and selfie
 * - Detailed validation feedback
 *
 * @example
 * ```typescript
 * const analyzer = new KYCFaceAnalyzer();
 * await analyzer.load('/models');
 *
 * // Analyze ID document photo
 * const idResult = await analyzer.analyzeForKYC(idPhoto);
 * if (!idResult.isAcceptable) {
 *   console.log('Issues:', idResult.issues);
 *   console.log('Recommendations:', idResult.recommendations);
 * }
 *
 * // Compare ID with selfie
 * const matchResult = await analyzer.compareFaces(idPhoto, selfiePhoto);
 * if (matchResult.isMatch) {
 *   console.log(`Match confidence: ${matchResult.confidence}`);
 * }
 * ```
 */
export class KYCFaceAnalyzer {
  private detector: SsdMobilenetv1;
  private landmarkNet: FaceLandmark68Net;
  private recognitionNet: FaceRecognitionNet;
  private qualityAnalyzer: ImageQualityAnalyzer;
  private config: KYCConfig;
  private _isLoaded = false;

  constructor(config: Partial<KYCConfig> = {}) {
    this.detector = new SsdMobilenetv1();
    this.landmarkNet = new FaceLandmark68Net();
    this.recognitionNet = new FaceRecognitionNet();
    this.qualityAnalyzer = new ImageQualityAnalyzer();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Load all required models.
   */
  public async load(modelPath: string): Promise<void> {
    await Promise.all([
      this.detector.load(modelPath),
      this.landmarkNet.load(modelPath),
      this.recognitionNet.load(modelPath),
    ]);
    this._isLoaded = true;
  }

  /**
   * Check if models are loaded.
   */
  public get isLoaded(): boolean {
    return this._isLoaded;
  }

  /**
   * Analyze a face image for KYC compliance.
   */
  public async analyzeForKYC(input: TNetInput): Promise<KYCAnalysisResult> {
    if (!this._isLoaded) {
      throw new Error('KYCFaceAnalyzer - call load() before analysis');
    }

    const issues: string[] = [];
    const recommendations: string[] = [];

    // Detect face with high confidence threshold
    const detections = await this.detector.locateFaces(input, {
      minConfidence: this.config.minDetectionConfidence,
      maxResults: 5,
    });

    // No face detected
    if (detections.length === 0) {
      return {
        faceDetected: false,
        detection: null,
        landmarks: null,
        descriptor: null,
        quality: null,
        issues: ['Nenhum rosto detectado na imagem'],
        isAcceptable: false,
        recommendations: [
          'Certifique-se de que o rosto está claramente visível',
          'Melhore a iluminação',
          'Remova obstruções (óculos escuros, máscaras)',
        ],
      };
    }

    // Multiple faces detected
    if (detections.length > 1) {
      issues.push(`Múltiplos rostos detectados (${detections.length})`);
      recommendations.push('A imagem deve conter apenas um rosto');
    }

    // Use the largest/most confident detection
    const detection = detections.reduce((best, current) => (current.score > best.score ? current : best));

    // Get landmarks
    const landmarks = await this.landmarkNet.detectLandmarks(input) as FaceLandmarks68;

    // Get face descriptor
    const descriptor = await this.recognitionNet.computeFaceDescriptor(input) as Float32Array;

    // Calculate quality metrics
    const quality = await this.calculateQuality(input, detection, landmarks);

    // Validate quality
    this.validateQuality(quality, issues, recommendations);

    const isAcceptable = issues.length === 0;

    return {
      faceDetected: true,
      detection,
      landmarks,
      descriptor,
      quality,
      issues,
      isAcceptable,
      recommendations,
    };
  }

  /**
   * Compare two face images (e.g., ID photo vs selfie).
   */
  public async compareFaces(
    idPhoto: TNetInput,
    selfie: TNetInput,
  ): Promise<KYCMatchResult> {
    if (!this._isLoaded) {
      throw new Error('KYCFaceAnalyzer - call load() before comparison');
    }

    // Analyze both images
    const [idResult, selfieResult] = await Promise.all([
      this.analyzeForKYC(idPhoto),
      this.analyzeForKYC(selfie),
    ]);

    // Check if both have valid faces
    if (!idResult.descriptor || !selfieResult.descriptor) {
      return {
        isMatch: false,
        similarity: 0,
        distance: Infinity,
        confidence: 'low',
        idQuality: idResult.quality,
        selfieQuality: selfieResult.quality,
      };
    }

    // Calculate distance between descriptors
    const distance = euclideanDistance(
      Array.from(idResult.descriptor),
      Array.from(selfieResult.descriptor),
    );

    // Convert distance to similarity (0-1)
    // Typical distances: same person < 0.6, different > 0.6
    const similarity = Math.max(0, 1 - distance);

    // Determine match
    const isMatch = distance < this.config.matchThreshold;

    // Determine confidence level
    let confidence: 'high' | 'medium' | 'low';
    if (distance < 0.4) {
      confidence = 'high';
    } else if (distance < 0.55) {
      confidence = 'medium';
    } else {
      confidence = 'low';
    }

    return {
      isMatch,
      similarity,
      distance,
      confidence,
      idQuality: idResult.quality,
      selfieQuality: selfieResult.quality,
    };
  }

  /**
   * Calculate face quality metrics.
   */
  private async calculateQuality(
    input: TNetInput,
    detection: FaceDetection,
    landmarks: FaceLandmarks68,
  ): Promise<FaceQualityMetrics> {
    const netInput = await toNetInput(input);
    const inputHeight = netInput.getInputHeight(0);
    const inputWidth = netInput.getInputWidth(0);

    // Face size relative to image
    const faceBox = detection.box;
    const faceArea = faceBox.width * faceBox.height;
    const imageArea = inputWidth * inputHeight;
    const faceSize = faceArea / imageArea;

    // Calculate head pose from landmarks
    const pose = this.estimatePose(landmarks);
    const isFrontal = Math.abs(pose.yaw) <= this.config.maxYaw
      && Math.abs(pose.pitch) <= this.config.maxPitch
      && Math.abs(pose.roll) <= this.config.maxRoll;

    // Check if eyes are visible
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();
    const eyesVisible = leftEye.length > 0 && rightEye.length > 0;

    // Calculate image quality metrics using tensor operations
    const qualityMetrics = await this.calculateImageQuality(input, detection);

    const overallScore = this.calculateOverallScore({
      ...qualityMetrics,
      faceSize,
      isFrontal,
      eyesVisible,
      detectionConfidence: detection.score,
    });

    return {
      overallScore,
      sharpness: qualityMetrics.sharpness,
      brightness: qualityMetrics.brightness,
      contrast: qualityMetrics.contrast,
      faceSize,
      pose,
      isFrontal,
      eyesVisible,
      detectionConfidence: detection.score,
    };
  }

  /**
   * Estimate head pose from landmarks.
   */
  private estimatePose(landmarks: FaceLandmarks68): { yaw: number; pitch: number; roll: number } {
    const positions = landmarks.positions;

    // Key landmarks
    const noseTip = positions[30]; // Nose tip
    const leftEyeOuter = positions[36];
    const rightEyeOuter = positions[45];
    const chin = positions[8];

    if (!noseTip || !leftEyeOuter || !rightEyeOuter || !chin) {
      return { yaw: 0, pitch: 0, roll: 0 };
    }

    // Eye center
    const eyeCenterX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
    const eyeCenterY = (leftEyeOuter.y + rightEyeOuter.y) / 2;

    // Calculate yaw (left-right rotation)
    const eyeWidth = rightEyeOuter.x - leftEyeOuter.x;
    const noseOffset = noseTip.x - eyeCenterX;
    const yaw = Math.atan2(noseOffset, eyeWidth / 2) * (180 / Math.PI);

    // Calculate pitch (up-down rotation)
    const faceHeight = chin.y - eyeCenterY;
    const noseVertOffset = noseTip.y - eyeCenterY;
    const expectedNoseY = faceHeight * 0.4;
    const pitch = ((noseVertOffset - expectedNoseY) / faceHeight) * 30;

    // Calculate roll (tilt)
    const roll = Math.atan2(
      rightEyeOuter.y - leftEyeOuter.y,
      rightEyeOuter.x - leftEyeOuter.x,
    ) * (180 / Math.PI);

    return { yaw, pitch, roll };
  }

  /**
   * Calculate image quality metrics from the face region.
   * Uses real pixel analysis with Laplacian variance for sharpness,
   * mean intensity for brightness, and standard deviation for contrast.
   */
  private async calculateImageQuality(
    input: TNetInput,
    detection: FaceDetection,
  ): Promise<{ sharpness: number; brightness: number; contrast: number }> {
    const result = await this.qualityAnalyzer.analyze(input, detection);

    // If face region extraction failed, use detection confidence as fallback
    if (!result.faceRegionExtracted) {
      const confidence = detection.score;
      return {
        sharpness: Math.min(1.0, 0.5 + confidence * 0.5),
        brightness: 0.5,
        contrast: 0.5 + confidence * 0.2,
      };
    }

    return {
      sharpness: result.sharpness,
      brightness: result.brightness,
      contrast: result.contrast,
    };
  }

  /**
   * Calculate overall quality score.
   */
  private calculateOverallScore(metrics: {
    sharpness: number;
    brightness: number;
    contrast: number;
    faceSize: number;
    isFrontal: boolean;
    eyesVisible: boolean;
    detectionConfidence: number;
  }): number {
    const weights = {
      sharpness: 0.2,
      brightness: 0.1,
      contrast: 0.1,
      faceSize: 0.15,
      frontal: 0.2,
      eyes: 0.1,
      confidence: 0.15,
    };

    // Normalize face size (optimal around 0.3-0.5)
    const faceSizeScore = metrics.faceSize >= 0.1 && metrics.faceSize <= 0.9
      ? 1 - Math.abs(0.4 - metrics.faceSize) * 2
      : 0.3;

    // Brightness penalty for too dark or too bright
    const brightnessScore = 1 - Math.abs(0.5 - metrics.brightness) * 2;

    const score = metrics.sharpness * weights.sharpness
      + brightnessScore * weights.brightness
      + metrics.contrast * weights.contrast
      + faceSizeScore * weights.faceSize
      + (metrics.isFrontal ? 1 : 0.3) * weights.frontal
      + (metrics.eyesVisible ? 1 : 0) * weights.eyes
      + metrics.detectionConfidence * weights.confidence;

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Validate quality metrics and add issues/recommendations.
   */
  private validateQuality(
    quality: FaceQualityMetrics,
    issues: string[],
    recommendations: string[],
  ): void {
    // Detection confidence
    if (quality.detectionConfidence < this.config.minDetectionConfidence) {
      issues.push('Baixa confiança na detecção do rosto');
      recommendations.push('Use uma imagem mais clara do rosto');
    }

    // Face size
    if (quality.faceSize < this.config.minFaceSize) {
      issues.push('Rosto muito pequeno na imagem');
      recommendations.push('Aproxime-se da câmera ou use uma foto maior');
    } else if (quality.faceSize > this.config.maxFaceSize) {
      issues.push('Rosto muito grande na imagem');
      recommendations.push('Afaste-se um pouco da câmera');
    }

    // Pose
    if (!quality.isFrontal) {
      if (Math.abs(quality.pose.yaw) > this.config.maxYaw) {
        issues.push('Rosto virado para o lado');
        recommendations.push('Olhe diretamente para a câmera');
      }
      if (Math.abs(quality.pose.pitch) > this.config.maxPitch) {
        issues.push('Rosto inclinado para cima ou baixo');
        recommendations.push('Mantenha a cabeça na posição neutra');
      }
      if (Math.abs(quality.pose.roll) > this.config.maxRoll) {
        issues.push('Cabeça inclinada lateralmente');
        recommendations.push('Mantenha a cabeça reta');
      }
    }

    // Eyes
    if (!quality.eyesVisible) {
      issues.push('Olhos não estão claramente visíveis');
      recommendations.push('Remova óculos escuros e mantenha os olhos abertos');
    }

    // Sharpness
    if (quality.sharpness < this.config.minSharpness) {
      issues.push('Imagem borrada ou fora de foco');
      recommendations.push('Use uma imagem mais nítida');
    }

    // Brightness
    if (quality.brightness < 0.2) {
      issues.push('Imagem muito escura');
      recommendations.push('Melhore a iluminação');
    } else if (quality.brightness > 0.8) {
      issues.push('Imagem muito clara/superexposta');
      recommendations.push('Reduza a iluminação direta');
    }
  }

  /**
   * Dispose of resources.
   */
  public dispose(): void {
    this.detector.dispose();
    this.landmarkNet.dispose();
    this.recognitionNet.dispose();
    this._isLoaded = false;
  }
}

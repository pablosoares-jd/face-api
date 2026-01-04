import { FaceDetection } from '../classes/FaceDetection';
import { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import { TNetInput, toNetInput } from '../dom/index';
import { SsdMobilenetv1 } from '../ssdMobilenetv1/SsdMobilenetv1';
import { FaceLandmark68Net } from '../faceLandmarkNet/FaceLandmark68Net';
import { FaceRecognitionNet } from '../faceRecognitionNet/FaceRecognitionNet';
import { euclideanDistance } from '../euclideanDistance';
import { ImageQualityAnalyzer } from './ImageQualityAnalyzer';

/**
 * Source type for the image being analyzed.
 */
export type ImageSourceType = 'selfie' | 'document' | 'unknown';

/**
 * Document types supported.
 */
export type DocumentType = 'rg' | 'cnh' | 'passport' | 'other';

/**
 * Quality thresholds that vary by source type.
 */
export interface AdaptiveThresholds {
  /** Minimum detection confidence */
  minConfidence: number;
  /** Minimum sharpness */
  minSharpness: number;
  /** Maximum yaw angle */
  maxYaw: number;
  /** Maximum pitch angle */
  maxPitch: number;
  /** Minimum face size */
  minFaceSize: number;
}

/**
 * Thresholds optimized for different image sources.
 */
const SOURCE_THRESHOLDS: Record<ImageSourceType, AdaptiveThresholds> = {
  selfie: {
    minConfidence: 0.85,
    minSharpness: 0.4,
    maxYaw: 12,
    maxPitch: 12,
    minFaceSize: 0.15,
  },
  document: {
    // More lenient for document photos
    minConfidence: 0.6,    // Documents often have lower detection confidence
    minSharpness: 0.2,     // Printed/scanned photos are often less sharp
    maxYaw: 20,            // Old photos may have slight angles
    maxPitch: 20,
    minFaceSize: 0.05,     // Document photos are often smaller
  },
  unknown: {
    minConfidence: 0.7,
    minSharpness: 0.3,
    maxYaw: 15,
    maxPitch: 15,
    minFaceSize: 0.1,
  },
};

/**
 * Analysis result for a specific image source.
 */
export interface SourceAnalysisResult {
  /** Type of source analyzed */
  sourceType: ImageSourceType;

  /** Was a face detected? */
  faceDetected: boolean;

  /** Detection result */
  detection: FaceDetection | null;

  /** Landmarks */
  landmarks: FaceLandmarks68 | null;

  /** Face descriptor */
  descriptor: Float32Array | null;

  /** Quality score (0-1) */
  qualityScore: number;

  /** Detailed quality breakdown */
  quality: {
    sharpness: number;
    brightness: number;
    contrast: number;
    faceSize: number;
    frontalScore: number;
    overallConfidence: number;
  };

  /** Estimated image issues */
  estimatedIssues: string[];

  /** Is acceptable for this source type? */
  isAcceptable: boolean;
}

/**
 * Complete KYC verification result.
 */
export interface KYCVerificationResult {
  /** Selfie analysis */
  selfie: SourceAnalysisResult;

  /** Document analysis */
  document: SourceAnalysisResult;

  /** Match result */
  match: {
    /** Final decision */
    isMatch: boolean;

    /** Raw distance between descriptors */
    distance: number;

    /** Similarity percentage (0-100) */
    similarityPercent: number;

    /** Adjusted threshold used (varies by quality) */
    thresholdUsed: number;

    /** Confidence level */
    confidence: 'very_high' | 'high' | 'medium' | 'low' | 'very_low';

    /** Risk score (0-100, lower = safer) */
    riskScore: number;
  };

  /** Overall verification status */
  status: 'approved' | 'needs_review' | 'rejected';

  /** Reasons for the decision */
  reasons: string[];

  /** Recommendations for the operator */
  operatorNotes: string[];
}

/**
 * Configuration for document KYC verification.
 */
export interface DocumentKYCConfig {
  /** Base match threshold (default: 0.6) */
  baseMatchThreshold: number;

  /** Allow quality-based threshold adjustment (default: true) */
  adaptiveThreshold: boolean;

  /** Maximum threshold adjustment (default: 0.15) */
  maxThresholdAdjustment: number;

  /** Require manual review for medium confidence (default: true) */
  requireReviewForMedium: boolean;

  /** Document type being verified */
  documentType: DocumentType;
}

const DEFAULT_CONFIG: DocumentKYCConfig = {
  baseMatchThreshold: 0.6,
  adaptiveThreshold: true,
  maxThresholdAdjustment: 0.15,
  requireReviewForMedium: true,
  documentType: 'other',
};

/**
 * KYC Document Analyzer - Specialized for ID document vs selfie verification.
 *
 * Handles real-world challenges:
 * - Document photos are often old, low quality, printed/scanned
 * - Selfies are current, higher quality, but may have different lighting
 * - Age difference between document and current appearance
 * - Different thresholds for document vs selfie quality
 * - Adaptive matching based on image quality
 *
 * @example
 * ```typescript
 * const analyzer = new KYCDocumentAnalyzer();
 * await analyzer.load('/models');
 *
 * const result = await analyzer.verifyIdentity(
 *   selfieImage,
 *   documentImage,
 *   { documentType: 'cnh' }
 * );
 *
 * if (result.status === 'approved') {
 *   console.log(`✅ Aprovado - Similaridade: ${result.match.similarityPercent}%`);
 * } else if (result.status === 'needs_review') {
 *   console.log('⚠️ Necessita revisão manual');
 *   console.log('Notas:', result.operatorNotes);
 * } else {
 *   console.log('❌ Rejeitado');
 *   console.log('Motivos:', result.reasons);
 * }
 * ```
 */
export class KYCDocumentAnalyzer {
  private detector: SsdMobilenetv1;
  private landmarkNet: FaceLandmark68Net;
  private recognitionNet: FaceRecognitionNet;
  private qualityAnalyzer: ImageQualityAnalyzer;
  private config: DocumentKYCConfig;
  private _isLoaded = false;

  constructor(config: Partial<DocumentKYCConfig> = {}) {
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
   * Analyze a selfie image.
   */
  public async analyzeSelfie(input: TNetInput): Promise<SourceAnalysisResult> {
    return this.analyzeImage(input, 'selfie');
  }

  /**
   * Analyze a document photo.
   */
  public async analyzeDocument(input: TNetInput): Promise<SourceAnalysisResult> {
    return this.analyzeImage(input, 'document');
  }

  /**
   * Complete identity verification: selfie vs document.
   */
  public async verifyIdentity(
    selfie: TNetInput,
    document: TNetInput,
    options: Partial<DocumentKYCConfig> = {},
  ): Promise<KYCVerificationResult> {
    if (!this._isLoaded) {
      throw new Error('KYCDocumentAnalyzer - call load() before verification');
    }

    const config = { ...this.config, ...options };

    // Analyze both images in parallel
    const [selfieResult, documentResult] = await Promise.all([
      this.analyzeSelfie(selfie),
      this.analyzeDocument(document),
    ]);

    const reasons: string[] = [];
    const operatorNotes: string[] = [];

    // Check if both have valid faces
    if (!selfieResult.faceDetected) {
      return this.createRejectionResult(
        selfieResult,
        documentResult,
        ['Nenhum rosto detectado na selfie'],
        ['Solicitar nova selfie com rosto claramente visível'],
      );
    }

    if (!documentResult.faceDetected) {
      return this.createRejectionResult(
        selfieResult,
        documentResult,
        ['Nenhum rosto detectado no documento'],
        ['Verificar qualidade da foto do documento', 'Solicitar nova foto do documento'],
      );
    }

    // Calculate adaptive threshold based on quality
    const effectiveThreshold = this.calculateAdaptiveThreshold(
      selfieResult,
      documentResult,
      config,
    );

    // Calculate match
    const distance = euclideanDistance(
      Array.from(selfieResult.descriptor!),
      Array.from(documentResult.descriptor!),
    );

    const similarityPercent = Math.max(0, (1 - distance) * 100);
    const isMatch = distance < effectiveThreshold;

    // Determine confidence level
    const confidence = this.determineConfidence(distance, effectiveThreshold, selfieResult, documentResult);

    // Calculate risk score
    const riskScore = this.calculateRiskScore(distance, effectiveThreshold, selfieResult, documentResult);

    // Determine final status
    let status: 'approved' | 'needs_review' | 'rejected';

    if (isMatch && confidence === 'very_high') {
      status = 'approved';
    } else if (isMatch && (confidence === 'high' || confidence === 'medium')) {
      status = config.requireReviewForMedium && confidence === 'medium'
        ? 'needs_review'
        : 'approved';
    } else if (isMatch && confidence === 'low') {
      status = 'needs_review';
    } else {
      status = 'rejected';
    }

    // Add quality-based notes
    if (documentResult.qualityScore < 0.5) {
      operatorNotes.push(`Qualidade do documento baixa (${(documentResult.qualityScore * 100).toFixed(0)}%)`);
      operatorNotes.push('Foto do documento pode ser antiga ou de baixa resolução');
    }

    if (selfieResult.qualityScore < 0.6) {
      operatorNotes.push(`Qualidade da selfie moderada (${(selfieResult.qualityScore * 100).toFixed(0)}%)`);
    }

    // Add distance-based notes
    if (distance > 0.5 && distance < effectiveThreshold) {
      operatorNotes.push('Similaridade próxima do limite - verificar manualmente');
    }

    // Document-specific notes
    if (config.documentType === 'rg' || config.documentType === 'cnh') {
      operatorNotes.push(`Documento: ${config.documentType.toUpperCase()}`);
      if (documentResult.qualityScore < 0.4) {
        operatorNotes.push('Considerar diferença de idade entre documento e selfie');
      }
    }

    // Add reasons for status
    if (status === 'approved') {
      reasons.push(`Similaridade: ${similarityPercent.toFixed(1)}%`);
      reasons.push(`Confiança: ${this.translateConfidence(confidence)}`);
    } else if (status === 'needs_review') {
      reasons.push('Verificação automática inconclusiva');
      if (confidence === 'medium' || confidence === 'low') {
        reasons.push(`Confiança ${this.translateConfidence(confidence)} requer revisão`);
      }
      if (documentResult.qualityScore < 0.5) {
        reasons.push('Qualidade do documento afeta precisão');
      }
    } else {
      reasons.push(`Similaridade insuficiente: ${similarityPercent.toFixed(1)}%`);
      reasons.push(`Distância: ${distance.toFixed(3)} (limite: ${effectiveThreshold.toFixed(3)})`);
    }

    return {
      selfie: selfieResult,
      document: documentResult,
      match: {
        isMatch,
        distance,
        similarityPercent,
        thresholdUsed: effectiveThreshold,
        confidence,
        riskScore,
      },
      status,
      reasons,
      operatorNotes,
    };
  }

  /**
   * Analyze an image with source-specific thresholds.
   */
  private async analyzeImage(
    input: TNetInput,
    sourceType: ImageSourceType,
  ): Promise<SourceAnalysisResult> {
    const thresholds = SOURCE_THRESHOLDS[sourceType];

    // Detect face with source-appropriate confidence
    const detections = await this.detector.locateFaces(input, {
      minConfidence: thresholds.minConfidence,
      maxResults: 3,
    });

    if (detections.length === 0) {
      return this.createEmptyResult(sourceType);
    }

    // Use best detection
    const detection = detections.reduce((best, current) =>
      current.score > best.score ? current : best
    );

    // Get landmarks and descriptor
    let landmarks: FaceLandmarks68 | null = null;
    let descriptor: Float32Array | null = null;

    try {
      landmarks = await this.landmarkNet.detectLandmarks(input) as FaceLandmarks68;
      descriptor = await this.recognitionNet.computeFaceDescriptor(input) as Float32Array;
    } catch {
      // Continue with what we have
    }

    // Calculate quality metrics
    const quality = await this.calculateQuality(input, detection, landmarks, sourceType);
    const qualityScore = this.calculateQualityScore(quality, sourceType);

    // Check acceptability
    const estimatedIssues = this.identifyIssues(quality, thresholds, sourceType);
    const isAcceptable = this.checkAcceptability(quality, thresholds, estimatedIssues);

    return {
      sourceType,
      faceDetected: true,
      detection,
      landmarks,
      descriptor,
      qualityScore,
      quality,
      estimatedIssues,
      isAcceptable,
    };
  }

  /**
   * Calculate quality metrics for an image.
   * Uses real pixel analysis with Laplacian variance for sharpness.
   */
  private async calculateQuality(
    input: TNetInput,
    detection: FaceDetection,
    landmarks: FaceLandmarks68 | null,
    sourceType: ImageSourceType,
  ): Promise<SourceAnalysisResult['quality']> {
    const netInput = await toNetInput(input);
    const inputHeight = netInput.getInputHeight(0);
    const inputWidth = netInput.getInputWidth(0);

    // Face size relative to image
    const faceBox = detection.box;
    const faceArea = faceBox.width * faceBox.height;
    const imageArea = inputWidth * inputHeight;
    const faceSize = faceArea / imageArea;

    // Frontal score from landmarks
    let frontalScore = 0.8;
    if (landmarks) {
      const pose = this.estimatePose(landmarks);
      const yawPenalty = Math.min(1, Math.abs(pose.yaw) / 45);
      const pitchPenalty = Math.min(1, Math.abs(pose.pitch) / 45);
      frontalScore = 1 - (yawPenalty * 0.5 + pitchPenalty * 0.5);
    }

    // Use real pixel analysis
    const qualityResult = await this.qualityAnalyzer.analyze(input, detection);

    // Apply source-type adjustments for expected quality
    let sharpness = qualityResult.sharpness;
    let brightness = qualityResult.brightness;
    let contrast = qualityResult.contrast;

    // Documents are expected to have lower quality, so we adjust expectations
    if (sourceType === 'document' && qualityResult.faceRegionExtracted) {
      // Boost sharpness score for documents (they're often scanned)
      sharpness = Math.min(1, sharpness * 1.2);
    }

    // Fallback if analysis failed
    if (!qualityResult.faceRegionExtracted) {
      const baseSharpness = sourceType === 'document' ? 0.5 : 0.7;
      sharpness = baseSharpness * (0.5 + detection.score * 0.5);
      brightness = 0.5;
      contrast = 0.5 + detection.score * 0.2;
    }

    return {
      sharpness,
      brightness,
      contrast,
      faceSize,
      frontalScore,
      overallConfidence: detection.score,
    };
  }

  /**
   * Estimate pose from landmarks.
   */
  private estimatePose(landmarks: FaceLandmarks68): { yaw: number; pitch: number; roll: number } {
    const positions = landmarks.positions;
    const noseTip = positions[30];
    const leftEye = positions[36];
    const rightEye = positions[45];

    if (!noseTip || !leftEye || !rightEye) {
      return { yaw: 0, pitch: 0, roll: 0 };
    }

    const eyeCenterX = (leftEye.x + rightEye.x) / 2;
    const eyeWidth = rightEye.x - leftEye.x;
    const noseOffset = noseTip.x - eyeCenterX;

    const yaw = Math.atan2(noseOffset, eyeWidth / 2) * (180 / Math.PI);
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * (180 / Math.PI);

    return { yaw, pitch: 0, roll };
  }

  /**
   * Calculate overall quality score.
   */
  private calculateQualityScore(
    quality: SourceAnalysisResult['quality'],
    sourceType: ImageSourceType,
  ): number {
    const weights = sourceType === 'document'
      ? { sharpness: 0.2, brightness: 0.1, contrast: 0.1, faceSize: 0.2, frontal: 0.2, confidence: 0.2 }
      : { sharpness: 0.25, brightness: 0.1, contrast: 0.1, faceSize: 0.15, frontal: 0.2, confidence: 0.2 };

    const faceSizeScore = quality.faceSize >= 0.05 && quality.faceSize <= 0.9
      ? Math.min(1, quality.faceSize * 5)
      : 0.3;

    return (
      quality.sharpness * weights.sharpness +
      quality.brightness * weights.brightness +
      quality.contrast * weights.contrast +
      faceSizeScore * weights.faceSize +
      quality.frontalScore * weights.frontal +
      quality.overallConfidence * weights.confidence
    );
  }

  /**
   * Identify issues with the image.
   */
  private identifyIssues(
    quality: SourceAnalysisResult['quality'],
    thresholds: AdaptiveThresholds,
    sourceType: ImageSourceType,
  ): string[] {
    const issues: string[] = [];
    const label = sourceType === 'selfie' ? 'Selfie' : 'Documento';

    if (quality.sharpness < thresholds.minSharpness) {
      issues.push(`${label}: Imagem pode estar borrada`);
    }

    if (quality.faceSize < thresholds.minFaceSize) {
      issues.push(`${label}: Rosto muito pequeno`);
    }

    if (quality.frontalScore < 0.6) {
      issues.push(`${label}: Rosto não está de frente`);
    }

    if (quality.overallConfidence < thresholds.minConfidence) {
      issues.push(`${label}: Baixa confiança na detecção`);
    }

    if (quality.brightness < 0.2) {
      issues.push(`${label}: Imagem muito escura`);
    } else if (quality.brightness > 0.8) {
      issues.push(`${label}: Imagem muito clara`);
    }

    return issues;
  }

  /**
   * Check if image meets acceptability criteria.
   */
  private checkAcceptability(
    quality: SourceAnalysisResult['quality'],
    thresholds: AdaptiveThresholds,
    issues: string[],
  ): boolean {
    // Must have minimum confidence and face size
    if (quality.overallConfidence < thresholds.minConfidence * 0.8) return false;
    if (quality.faceSize < thresholds.minFaceSize * 0.5) return false;

    // Allow some issues for documents
    return issues.length <= 2;
  }

  /**
   * Calculate adaptive threshold based on image quality.
   */
  private calculateAdaptiveThreshold(
    selfieResult: SourceAnalysisResult,
    documentResult: SourceAnalysisResult,
    config: DocumentKYCConfig,
  ): number {
    if (!config.adaptiveThreshold) {
      return config.baseMatchThreshold;
    }

    let threshold = config.baseMatchThreshold;

    // Relax threshold if document quality is low
    if (documentResult.qualityScore < 0.5) {
      const qualityPenalty = (0.5 - documentResult.qualityScore) * 0.2;
      threshold += Math.min(qualityPenalty, config.maxThresholdAdjustment);
    }

    // Slightly relax if selfie quality is also not great
    if (selfieResult.qualityScore < 0.6) {
      threshold += 0.03;
    }

    // Cap the adjustment
    return Math.min(threshold, config.baseMatchThreshold + config.maxThresholdAdjustment);
  }

  /**
   * Determine confidence level.
   */
  private determineConfidence(
    distance: number,
    threshold: number,
    selfieResult: SourceAnalysisResult,
    documentResult: SourceAnalysisResult,
  ): 'very_high' | 'high' | 'medium' | 'low' | 'very_low' {
    const _margin = threshold - distance; // Reserved for future use
    const avgQuality = (selfieResult.qualityScore + documentResult.qualityScore) / 2;

    if (distance < 0.35 && avgQuality > 0.7) return 'very_high';
    if (distance < 0.45 && avgQuality > 0.6) return 'high';
    if (distance < 0.55) return 'medium';
    if (distance < threshold) return 'low';
    return 'very_low';
  }

  /**
   * Calculate risk score (0-100).
   */
  private calculateRiskScore(
    distance: number,
    threshold: number,
    selfieResult: SourceAnalysisResult,
    documentResult: SourceAnalysisResult,
  ): number {
    // Base risk from distance
    const distanceRisk = Math.min(100, (distance / threshold) * 60);

    // Quality risk
    const qualityRisk = (1 - (selfieResult.qualityScore + documentResult.qualityScore) / 2) * 30;

    // Issues risk
    const issuesRisk = (selfieResult.estimatedIssues.length + documentResult.estimatedIssues.length) * 5;

    return Math.min(100, distanceRisk + qualityRisk + issuesRisk);
  }

  /**
   * Translate confidence level to Portuguese.
   */
  private translateConfidence(confidence: string): string {
    const translations: Record<string, string> = {
      very_high: 'Muito Alta',
      high: 'Alta',
      medium: 'Média',
      low: 'Baixa',
      very_low: 'Muito Baixa',
    };
    return translations[confidence] || confidence;
  }

  /**
   * Create empty result for failed detection.
   */
  private createEmptyResult(sourceType: ImageSourceType): SourceAnalysisResult {
    return {
      sourceType,
      faceDetected: false,
      detection: null,
      landmarks: null,
      descriptor: null,
      qualityScore: 0,
      quality: {
        sharpness: 0,
        brightness: 0,
        contrast: 0,
        faceSize: 0,
        frontalScore: 0,
        overallConfidence: 0,
      },
      estimatedIssues: ['Nenhum rosto detectado'],
      isAcceptable: false,
    };
  }

  /**
   * Create rejection result.
   */
  private createRejectionResult(
    selfieResult: SourceAnalysisResult,
    documentResult: SourceAnalysisResult,
    reasons: string[],
    notes: string[],
  ): KYCVerificationResult {
    return {
      selfie: selfieResult,
      document: documentResult,
      match: {
        isMatch: false,
        distance: Infinity,
        similarityPercent: 0,
        thresholdUsed: this.config.baseMatchThreshold,
        confidence: 'very_low',
        riskScore: 100,
      },
      status: 'rejected',
      reasons,
      operatorNotes: notes,
    };
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

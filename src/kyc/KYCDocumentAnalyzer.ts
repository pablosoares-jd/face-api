import * as tf from '@tensorflow/tfjs';
import * as blazefaceOfficial from '@tensorflow-models/blazeface';
import { FaceDetection } from '../classes/FaceDetection';
import type { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import { Rect } from '../classes/Rect';
import type { TNetInput } from '../dom/index';
import { toNetInput } from '../dom/index';
import type { BlazeFaceKeypoints } from '../blazeFace/BlazeFace';
import { BlazeFace, BlazeFaceDetection } from '../blazeFace/BlazeFace';
import { FaceLandmark68Net } from '../faceLandmarkNet/FaceLandmark68Net';
import { AdaFace } from '../adaFace/AdaFace';
import { TinyFaceDetector } from '../tinyFaceDetector/TinyFaceDetector';
import { euclideanDistance } from '../euclideanDistance';
import { ImageQualityAnalyzer } from './ImageQualityAnalyzer';

// Type for official BlazeFace model
type OfficialBlazeFaceModel = Awaited<ReturnType<typeof blazefaceOfficial.load>>;

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
    // Very lenient for document photos - faces are typically small
    minConfidence: 0.3, // Very low - document faces have low detection scores
    minSharpness: 0.15, // Printed/scanned photos are often less sharp
    maxYaw: 25, // Old photos may have slight angles
    maxPitch: 25,
    minFaceSize: 0.02, // Document faces can be as small as 2% of image
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

  /** Landmarks (68-point) */
  landmarks: FaceLandmarks68 | null;

  /** BlazeFace keypoints (6-point: eyes, ears, nose, mouth) */
  keypoints: BlazeFaceKeypoints | null;

  /** Face descriptor (512-dim for AdaFace, 128-dim for FaceNet fallback) */
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

  /** Model info for debugging */
  modelInfo: {
    detector: 'blazeface' | 'ssd_mobilenetv1' | 'tiny_face_detector';
    recognizer: 'adaface' | 'facenet';
    descriptorDim: number;
  };
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
 * Uses state-of-the-art models:
 * - **BlazeFace**: Ultra-fast detector (~98% accuracy, 200-1000+ FPS)
 * - **AdaFace**: Adaptive face recognition (99.82%+ LFW accuracy)
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
  private detector: BlazeFace;
  private tinyDetector: TinyFaceDetector;
  private landmarkNet: FaceLandmark68Net;
  private recognitionNet: AdaFace;
  private qualityAnalyzer: ImageQualityAnalyzer;
  private config: DocumentKYCConfig;
  private _isLoaded = false;
  private _officialBlazeFace: OfficialBlazeFaceModel | null = null;

  constructor(config: Partial<DocumentKYCConfig> = {}) {
    this.detector = new BlazeFace();
    this.tinyDetector = new TinyFaceDetector();
    this.landmarkNet = new FaceLandmark68Net();
    this.recognitionNet = new AdaFace();
    this.qualityAnalyzer = new ImageQualityAnalyzer();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Check if official BlazeFace is loaded.
   */
  public get isOfficialBlazeFaceLoaded(): boolean {
    return this._officialBlazeFace !== null;
  }

  /**
   * Load all required models.
   * Also loads the official BlazeFace model from TensorFlow Hub for better small face detection.
   */
  public async load(modelPath: string): Promise<void> {
    // Load local models in parallel
    const localModelsPromise = Promise.all([
      this.detector.load(modelPath),
      this.tinyDetector.load(modelPath),
      this.landmarkNet.load(modelPath),
      this.recognitionNet.load(modelPath),
    ]);

    // Try to load official BlazeFace (from TensorFlow Hub CDN)
    const officialBlazeFacePromise = blazefaceOfficial.load({
      maxFaces: 5,
      scoreThreshold: 0.5,
      iouThreshold: 0.3,
    }).then((model) => {
      this._officialBlazeFace = model;
      console.info('Official BlazeFace loaded from TensorFlow Hub');
    }).catch((err) => {
      console.warn('Failed to load official BlazeFace, using fallback detectors:', err);
    });

    // Wait for local models (required) and official BlazeFace (optional)
    await Promise.all([localModelsPromise, officialBlazeFacePromise]);
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

    // Check if descriptors were computed (face detected but recognition might have failed)
    if (!selfieResult.descriptor) {
      return this.createRejectionResult(
        selfieResult,
        documentResult,
        ['Não foi possível extrair características faciais da selfie'],
        ['Solicitar nova selfie com melhor iluminação e foco'],
      );
    }

    if (!documentResult.descriptor) {
      return this.createRejectionResult(
        selfieResult,
        documentResult,
        ['Não foi possível extrair características faciais do documento'],
        ['Verificar qualidade da foto do documento', 'Solicitar nova foto do documento'],
      );
    }

    // Calculate adaptive threshold based on quality
    const effectiveThreshold = this.calculateAdaptiveThreshold(
      selfieResult,
      documentResult,
      config,
    );

    // Calculate match - descriptors are guaranteed to be non-null here
    const distance = euclideanDistance(
      Array.from(selfieResult.descriptor),
      Array.from(documentResult.descriptor),
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
   * Uses BlazeFace for detection and AdaFace for recognition.
   */
  private async analyzeImage(
    input: TNetInput,
    sourceType: ImageSourceType,
  ): Promise<SourceAnalysisResult> {
    const thresholds = SOURCE_THRESHOLDS[sourceType];

    // Detect face with BlazeFace (ultra-fast, ~98% accuracy)
    // Type as FaceDetection[] since we may use fallback detectors that return FaceDetection
    let detections: FaceDetection[] = await this.detector.locateFaces(input, {
      minConfidence: thresholds.minConfidence,
      maxResults: 3,
      enableFallback: true, // Fall back to SSD MobileNetv1 if needed
    });

    // For documents, try multiple fallback strategies if no face found
    let usedTinyDetector = false;
    let usedOfficialBlazeFace = false;
    if (detections.length === 0 && sourceType === 'document') {
      // Strategy 0: Try Official BlazeFace from TensorFlow Hub (best for small faces)
      if (this._officialBlazeFace) {
        const officialDetections = await this.detectWithOfficialBlazeFace(input);
        if (officialDetections.length > 0) {
          detections = officialDetections;
          usedOfficialBlazeFace = true;
        }
      }

      // Strategy 1: Try TinyFaceDetector with maximum input size (608px)
      // Larger inputSize processes image at higher resolution = better small face detection
      if (detections.length === 0 && this.tinyDetector.isLoaded) {
        const tinyDetections = await this.tinyDetector.locateFaces(input, {
          scoreThreshold: 0.2, // Very low threshold
          inputSize: 608, // Maximum input size for best small face detection
        });
        if (tinyDetections.length > 0) {
          detections = tinyDetections;
          usedTinyDetector = true;
        }
      }

      // Strategy 2: If still no face, try SSD with ultra-low confidence
      if (detections.length === 0 && this.detector.isLoaded) {
        const ultraLowConfDetections = await this.detector.locateFaces(input, {
          minConfidence: 0.1, // Ultra-low confidence for difficult images
          maxResults: 5,
          enableFallback: true,
        });
        if (ultraLowConfDetections.length > 0) {
          detections = ultraLowConfDetections;
        }
      }

      // Strategy 3: Upscale image and try again (for very small faces in documents)
      if (detections.length === 0) {
        const upscaledResult = await this.detectWithUpscale(input);
        if (upscaledResult.detections.length > 0) {
          detections = upscaledResult.detections;
          usedTinyDetector = upscaledResult.usedTiny;
        }
      }
    }

    if (detections.length === 0) {
      return this.createEmptyResult(sourceType);
    }

    // Use best detection (highest score)
    const detection = detections.reduce((best, current) => (current.score > best.score ? current : best));

    // Extract BlazeFace keypoints if available
    let keypoints: BlazeFaceKeypoints | null = null;
    if (detection instanceof BlazeFaceDetection) {
      keypoints = detection.keypoints;
    }

    // Determine which detector was actually used
    let detectorUsed: 'blazeface' | 'ssd_mobilenetv1' | 'tiny_face_detector';
    if (usedOfficialBlazeFace) {
      detectorUsed = 'blazeface'; // Official BlazeFace from TensorFlow Hub
    } else if (usedTinyDetector) {
      detectorUsed = 'tiny_face_detector';
    } else if (this.detector.isPrimaryLoaded) {
      detectorUsed = 'blazeface';
    } else {
      detectorUsed = 'ssd_mobilenetv1';
    }

    // Get landmarks and descriptor
    let landmarks: FaceLandmarks68 | null = null;
    let descriptor: Float32Array | null = null;
    let recognizerUsed: 'adaface' | 'facenet' = 'adaface';

    try {
      landmarks = await this.landmarkNet.detectLandmarks(input) as FaceLandmarks68;
      descriptor = await this.recognitionNet.computeFaceDescriptor(input) as Float32Array;

      // Check if AdaFace or FaceNet fallback was used
      recognizerUsed = this.recognitionNet.isPrimaryLoaded ? 'adaface' : 'facenet';
    } catch {
      // Continue with what we have
    }

    // Calculate quality metrics (use keypoints for better pose estimation)
    const quality = await this.calculateQuality(input, detection, landmarks, sourceType, keypoints);
    const qualityScore = this.calculateQualityScore(quality, sourceType);

    // Check acceptability
    const estimatedIssues = this.identifyIssues(quality, thresholds, sourceType);
    const isAcceptable = this.checkAcceptability(quality, thresholds, estimatedIssues);

    return {
      sourceType,
      faceDetected: true,
      detection,
      landmarks,
      keypoints,
      descriptor,
      qualityScore,
      quality,
      estimatedIssues,
      isAcceptable,
      modelInfo: {
        detector: detectorUsed,
        recognizer: recognizerUsed,
        descriptorDim: descriptor?.length ?? 0,
      },
    };
  }

  /**
   * Detect faces with image upscaling (for very small faces in documents).
   * Upscales the image 2x-3x to make small faces more detectable.
   */
  private async detectWithUpscale(
    input: TNetInput,
  ): Promise<{ detections: FaceDetection[]; usedTiny: boolean }> {
    const netInput = await toNetInput(input);
    const inputElement = netInput.canvases[0] ?? netInput.getInput(0);

    if (!inputElement || inputElement instanceof tf.Tensor) {
      return { detections: [], usedTiny: false };
    }
    const canvas = inputElement as HTMLCanvasElement | HTMLImageElement;

    // Get original dimensions
    const originalWidth = canvas instanceof HTMLVideoElement
      ? canvas.videoWidth
      : canvas instanceof HTMLImageElement
        ? canvas.naturalWidth
        : canvas.width;
    const originalHeight = canvas instanceof HTMLVideoElement
      ? canvas.videoHeight
      : canvas instanceof HTMLImageElement
        ? canvas.naturalHeight
        : canvas.height;

    // Try upscale factors 2x and 3x
    for (const scale of [2, 3]) {
      const upscaledCanvas = document.createElement('canvas');
      upscaledCanvas.width = originalWidth * scale;
      upscaledCanvas.height = originalHeight * scale;
      const ctx = upscaledCanvas.getContext('2d');

      if (!ctx) continue;

      // Use high-quality image scaling
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(canvas as CanvasImageSource, 0, 0, upscaledCanvas.width, upscaledCanvas.height);

      // Try TinyFaceDetector first (best for small faces)
      if (this.tinyDetector.isLoaded) {
        const tinyDetections = await this.tinyDetector.locateFaces(upscaledCanvas, {
          scoreThreshold: 0.15, // Even lower for upscaled images
          inputSize: 608,
        });

        if (tinyDetections.length > 0) {
          // Adjust bounding boxes back to original scale
          const scaledDetections = tinyDetections.map((det) => {
            const box = det.relativeBox;
            return new FaceDetection(
              det.score,
              box, // relativeBox stays the same since it's relative to image
              { width: originalWidth, height: originalHeight },
            );
          });
          return { detections: scaledDetections, usedTiny: true };
        }
      }

      // Try SSD as fallback
      if (this.detector.isLoaded) {
        const ssdDetections = await this.detector.locateFaces(upscaledCanvas, {
          minConfidence: 0.08, // Very low confidence
          maxResults: 5,
          enableFallback: true,
        });

        if (ssdDetections.length > 0) {
          // Adjust bounding boxes back to original scale
          const scaledDetections = ssdDetections.map((det) => {
            const box = det.relativeBox;
            return new FaceDetection(
              det.score,
              box,
              { width: originalWidth, height: originalHeight },
            );
          });
          return { detections: scaledDetections, usedTiny: false };
        }
      }
    }

    return { detections: [], usedTiny: false };
  }

  /**
   * Detect faces using the official BlazeFace model from TensorFlow Hub.
   * This model is optimized for detecting faces of various sizes.
   */
  private async detectWithOfficialBlazeFace(input: TNetInput): Promise<FaceDetection[]> {
    if (!this._officialBlazeFace) {
      return [];
    }

    try {
      const netInput = await toNetInput(input);
      const inputElement = netInput.canvases[0] ?? netInput.getInput(0);

      if (!inputElement || inputElement instanceof tf.Tensor) {
        return [];
      }
      const canvas = inputElement as HTMLCanvasElement | HTMLImageElement | HTMLVideoElement;

      // Get image dimensions
      const width = canvas instanceof HTMLVideoElement
        ? canvas.videoWidth
        : canvas instanceof HTMLImageElement
          ? canvas.naturalWidth
          : canvas.width;
      const height = canvas instanceof HTMLVideoElement
        ? canvas.videoHeight
        : canvas instanceof HTMLImageElement
          ? canvas.naturalHeight
          : canvas.height;

      if (width === 0 || height === 0) {
        console.warn('Official BlazeFace: Invalid image dimensions');
        return [];
      }

      // Run official BlazeFace detection
      const predictions = await this._officialBlazeFace.estimateFaces(
        canvas as HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
        false, // returnTensors = false
      );

      if (!predictions || predictions.length === 0) {
        return [];
      }

      // Convert predictions to FaceDetection objects
      const detections: FaceDetection[] = [];

      for (const prediction of predictions) {
        try {
          // Handle both Tensor and array outputs for topLeft/bottomRight
          // Using async data() instead of blocking dataSync() for GPU efficiency
          let topLeftArr: number[];
          let bottomRightArr: number[];

          // Check if topLeft is a Tensor (has data method for async extraction)
          if (prediction.topLeft && typeof (prediction.topLeft as { data?: () => Promise<Float32Array | Int32Array | Uint8Array> }).data === 'function') {
            const tensorData = await (prediction.topLeft as { data: () => Promise<Float32Array | Int32Array | Uint8Array> }).data();
            topLeftArr = Array.from(tensorData);
          } else if (Array.isArray(prediction.topLeft)) {
            topLeftArr = prediction.topLeft as number[];
          } else {
            continue; // Skip invalid prediction
          }

          // Check if bottomRight is a Tensor
          if (prediction.bottomRight && typeof (prediction.bottomRight as { data?: () => Promise<Float32Array | Int32Array | Uint8Array> }).data === 'function') {
            const tensorData = await (prediction.bottomRight as { data: () => Promise<Float32Array | Int32Array | Uint8Array> }).data();
            bottomRightArr = Array.from(tensorData);
          } else if (Array.isArray(prediction.bottomRight)) {
            bottomRightArr = prediction.bottomRight as number[];
          } else {
            continue; // Skip invalid prediction
          }

          // Validate coordinates
          if (topLeftArr.length < 2 || bottomRightArr.length < 2) {
            continue;
          }

          const x = topLeftArr[0] ?? 0;
          const y = topLeftArr[1] ?? 0;
          const w = (bottomRightArr[0] ?? 0) - x;
          const h = (bottomRightArr[1] ?? 0) - y;

          // Skip invalid boxes
          if (w <= 0 || h <= 0) {
            continue;
          }

          // Get probability - handle both Tensor and number/array
          // Using async data() instead of blocking dataSync()
          let score = 0.5;
          if (prediction.probability) {
            if (typeof (prediction.probability as { data?: () => Promise<Float32Array | Int32Array | Uint8Array> }).data === 'function') {
              const probData = await (prediction.probability as { data: () => Promise<Float32Array | Int32Array | Uint8Array> }).data();
              score = probData[0] ?? 0.5;
            } else if (Array.isArray(prediction.probability)) {
              score = prediction.probability[0] ?? 0.5;
            } else if (typeof prediction.probability === 'number') {
              score = prediction.probability;
            }
          }

          // Create relative box (normalized coordinates)
          const relativeBox = new Rect(
            x / width,
            y / height,
            w / width,
            h / height,
          );

          detections.push(new FaceDetection(
            score,
            relativeBox,
            { width, height },
          ));
        } catch (predErr) {
          // Skip this prediction if conversion fails
          console.info('Skipping invalid prediction:', predErr);
        }
      }

      return detections;
    } catch (err) {
      console.warn('Official BlazeFace detection failed:', err);
      return [];
    }
  }

  /**
   * Calculate quality metrics for an image.
   * Uses real pixel analysis with Laplacian variance for sharpness.
   * Leverages BlazeFace keypoints for more accurate pose estimation when available.
   */
  private async calculateQuality(
    input: TNetInput,
    detection: FaceDetection,
    landmarks: FaceLandmarks68 | null,
    sourceType: ImageSourceType,
    keypoints: BlazeFaceKeypoints | null = null,
  ): Promise<SourceAnalysisResult['quality']> {
    const netInput = await toNetInput(input);
    const inputHeight = netInput.getInputHeight(0);
    const inputWidth = netInput.getInputWidth(0);

    // Face size relative to image
    const faceBox = detection.box;
    const faceArea = faceBox.width * faceBox.height;
    const imageArea = inputWidth * inputHeight;
    const faceSize = faceArea / imageArea;

    // Frontal score - prefer BlazeFace keypoints (6-point) over 68-point landmarks
    let frontalScore = 0.8;
    if (keypoints) {
      // BlazeFace keypoints are more reliable for pose estimation
      const pose = this.estimatePoseFromKeypoints(keypoints);
      const yawPenalty = Math.min(1, Math.abs(pose.yaw) / 45);
      const pitchPenalty = Math.min(1, Math.abs(pose.pitch) / 45);
      frontalScore = 1 - (yawPenalty * 0.5 + pitchPenalty * 0.5);
    } else if (landmarks) {
      // Fallback to 68-point landmarks
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
   * Estimate pose from BlazeFace keypoints (6-point).
   * More accurate than using 68-point landmarks for pose estimation.
   */
  private estimatePoseFromKeypoints(keypoints: BlazeFaceKeypoints): { yaw: number; pitch: number; roll: number } {
    const { leftEye, rightEye, noseTip, leftEar, rightEar } = keypoints;

    // Yaw from nose position relative to eye center
    const eyeCenterX = (leftEye.x + rightEye.x) / 2;
    const eyeWidth = Math.abs(rightEye.x - leftEye.x);
    const noseOffset = noseTip.x - eyeCenterX;
    const yaw = Math.atan2(noseOffset, eyeWidth / 2) * (180 / Math.PI);

    // Roll from eye alignment
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * (180 / Math.PI);

    // Pitch estimation using ear-to-eye ratio
    // When looking up/down, ear positions change relative to eyes
    const earCenterY = (leftEar.y + rightEar.y) / 2;
    const eyeCenterY = (leftEye.y + rightEye.y) / 2;
    const earEyeOffset = earCenterY - eyeCenterY;
    const faceHeight = eyeWidth * 1.5; // Approximate face height
    const pitch = Math.atan2(earEyeOffset, faceHeight / 2) * (180 / Math.PI) * 0.5;

    return { yaw, pitch, roll };
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
      quality.sharpness * weights.sharpness
      + quality.brightness * weights.brightness
      + quality.contrast * weights.contrast
      + faceSizeScore * weights.faceSize
      + quality.frontalScore * weights.frontal
      + quality.overallConfidence * weights.confidence
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
    // margin = threshold - distance reserved for future use
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
      keypoints: null,
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
      modelInfo: {
        detector: this.detector.isPrimaryLoaded ? 'blazeface' : 'ssd_mobilenetv1',
        recognizer: this.recognitionNet.isPrimaryLoaded ? 'adaface' : 'facenet',
        descriptorDim: 0,
      },
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

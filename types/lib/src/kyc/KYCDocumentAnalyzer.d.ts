import { FaceDetection } from '../classes/FaceDetection';
import { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import { TNetInput } from '../dom/index';
import { BlazeFaceKeypoints } from '../blazeFace/BlazeFace';
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
export declare class KYCDocumentAnalyzer {
    private detector;
    private tinyDetector;
    private landmarkNet;
    private recognitionNet;
    private qualityAnalyzer;
    private config;
    private _isLoaded;
    private _officialBlazeFace;
    constructor(config?: Partial<DocumentKYCConfig>);
    /**
     * Check if official BlazeFace is loaded.
     */
    get isOfficialBlazeFaceLoaded(): boolean;
    /**
     * Load all required models.
     * Also loads the official BlazeFace model from TensorFlow Hub for better small face detection.
     */
    load(modelPath: string): Promise<void>;
    /**
     * Check if models are loaded.
     */
    get isLoaded(): boolean;
    /**
     * Analyze a selfie image.
     */
    analyzeSelfie(input: TNetInput): Promise<SourceAnalysisResult>;
    /**
     * Analyze a document photo.
     */
    analyzeDocument(input: TNetInput): Promise<SourceAnalysisResult>;
    /**
     * Complete identity verification: selfie vs document.
     */
    verifyIdentity(selfie: TNetInput, document: TNetInput, options?: Partial<DocumentKYCConfig>): Promise<KYCVerificationResult>;
    /**
     * Analyze an image with source-specific thresholds.
     * Uses BlazeFace for detection and AdaFace for recognition.
     */
    private analyzeImage;
    /**
     * Detect faces with image upscaling (for very small faces in documents).
     * Upscales the image 2x-3x to make small faces more detectable.
     */
    private detectWithUpscale;
    /**
     * Detect faces using the official BlazeFace model from TensorFlow Hub.
     * This model is optimized for detecting faces of various sizes.
     */
    private detectWithOfficialBlazeFace;
    /**
     * Calculate quality metrics for an image.
     * Uses real pixel analysis with Laplacian variance for sharpness.
     * Leverages BlazeFace keypoints for more accurate pose estimation when available.
     */
    private calculateQuality;
    /**
     * Estimate pose from BlazeFace keypoints (6-point).
     * More accurate than using 68-point landmarks for pose estimation.
     */
    private estimatePoseFromKeypoints;
    /**
     * Estimate pose from landmarks.
     */
    private estimatePose;
    /**
     * Calculate overall quality score.
     */
    private calculateQualityScore;
    /**
     * Identify issues with the image.
     */
    private identifyIssues;
    /**
     * Check if image meets acceptability criteria.
     */
    private checkAcceptability;
    /**
     * Calculate adaptive threshold based on image quality.
     */
    private calculateAdaptiveThreshold;
    /**
     * Determine confidence level.
     */
    private determineConfidence;
    /**
     * Calculate risk score (0-100).
     */
    private calculateRiskScore;
    /**
     * Translate confidence level to Portuguese.
     */
    private translateConfidence;
    /**
     * Create empty result for failed detection.
     */
    private createEmptyResult;
    /**
     * Create rejection result.
     */
    private createRejectionResult;
    /**
     * Dispose of resources.
     */
    dispose(): void;
}

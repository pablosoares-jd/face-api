import type { FaceDetection } from '../classes/FaceDetection';
import type { FaceLandmarks68 } from '../classes/FaceLandmarks68';
import type { TNetInput } from '../dom/index';
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
        yaw: number;
        pitch: number;
        roll: number;
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
export declare class KYCFaceAnalyzer {
    private detector;
    private landmarkNet;
    private recognitionNet;
    private qualityAnalyzer;
    private config;
    private _isLoaded;
    constructor(config?: Partial<KYCConfig>);
    /**
     * Load all required models.
     */
    load(modelPath: string): Promise<void>;
    /**
     * Check if models are loaded.
     */
    get isLoaded(): boolean;
    /**
     * Analyze a face image for KYC compliance.
     */
    analyzeForKYC(input: TNetInput): Promise<KYCAnalysisResult>;
    /**
     * Compare two face images (e.g., ID photo vs selfie).
     */
    compareFaces(idPhoto: TNetInput, selfie: TNetInput): Promise<KYCMatchResult>;
    /**
     * Calculate face quality metrics.
     */
    private calculateQuality;
    /**
     * Estimate head pose from landmarks.
     */
    private estimatePose;
    /**
     * Calculate image quality metrics from the face region.
     * Uses real pixel analysis with Laplacian variance for sharpness,
     * mean intensity for brightness, and standard deviation for contrast.
     */
    private calculateImageQuality;
    /**
     * Calculate overall quality score.
     */
    private calculateOverallScore;
    /**
     * Validate quality metrics and add issues/recommendations.
     */
    private validateQuality;
    /**
     * Dispose of resources.
     */
    dispose(): void;
}

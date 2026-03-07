import { FaceDetection } from '../classes/FaceDetection';
import { TNetInput } from '../dom/index';
/**
 * Image quality metrics result.
 */
export interface ImageQualityResult {
    /** Sharpness score (0-1, higher = sharper) */
    sharpness: number;
    /** Brightness score (0-1, 0.5 = optimal) */
    brightness: number;
    /** Contrast score (0-1, higher = more contrast) */
    contrast: number;
    /** Whether the face region was successfully extracted */
    faceRegionExtracted: boolean;
}
/**
 * Image Quality Analyzer using real pixel analysis.
 *
 * Implements:
 * - Laplacian variance for sharpness/blur detection
 * - Mean pixel intensity for brightness
 * - Standard deviation for contrast
 */
export declare class ImageQualityAnalyzer {
    /**
     * Analyze image quality for the face region.
     */
    analyze(input: TNetInput, detection: FaceDetection): Promise<ImageQualityResult>;
    /**
     * Convert RGB tensor to grayscale.
     */
    private toGrayscale;
    /**
     * Extract face region from grayscale image.
     */
    private extractFaceRegion;
    /**
     * Calculate sharpness using Laplacian variance.
     * Higher variance = sharper image.
     */
    private calculateSharpness;
    /**
     * Calculate brightness as mean pixel intensity.
     */
    private calculateBrightness;
    /**
     * Calculate contrast as standard deviation of pixel intensities.
     */
    private calculateContrast;
    /**
     * Get default result when analysis fails.
     */
    private getDefaultResult;
}
/**
 * Singleton instance for convenience.
 */
export declare const imageQualityAnalyzer: ImageQualityAnalyzer;

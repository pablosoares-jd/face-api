import type { FaceDetection } from '../classes/FaceDetection';
import type { TNetInput } from '../dom/index';
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
 *
 * All tensor operations use async data() instead of blocking dataSync()
 * for better GPU pipeline efficiency.
 */
export declare class ImageQualityAnalyzer {
    /**
     * Analyze image quality for the face region.
     */
    analyze(input: TNetInput, detection: FaceDetection): Promise<ImageQualityResult>;
    /**
     * Convert RGB tensor to grayscale.
     * Assumes input is in [0, 255] range and normalizes to [0, 1].
     */
    private toGrayscale;
    /**
     * Extract face region from grayscale image.
     */
    private extractFaceRegion;
    /**
     * Calculate sharpness tensor using Laplacian variance.
     * Returns variance tensor for async extraction.
     * Higher variance = sharper image.
     */
    private calculateSharpnessTensor;
    /**
     * Calculate contrast tensor as variance of pixel intensities.
     * Returns variance tensor for async extraction.
     */
    private calculateContrastTensor;
    /**
     * Get default result when analysis fails.
     */
    private getDefaultResult;
}
/**
 * Singleton instance for convenience.
 */
export declare const imageQualityAnalyzer: ImageQualityAnalyzer;

/**
 * Shared constants for face-api neural networks.
 * Centralizes magic numbers and configuration values.
 */

/**
 * Mean RGB values used for image normalization in face feature extraction.
 * Used by: FaceFeatureExtractor, TinyFaceFeatureExtractor, TinyXception, FaceRecognitionNet
 *
 * These values are derived from the ImageNet dataset training statistics.
 */
export const FACE_FEATURE_MEAN_RGB: [number, number, number] = [122.782, 117.001, 104.298];

/**
 * Mean RGB values used for TinyFaceDetector and TinyYOLOv2 with separable convolutions.
 * Different from FACE_FEATURE_MEAN_RGB due to different model training.
 */
export const TINY_DETECTOR_MEAN_RGB: [number, number, number] = [117.001, 114.697, 97.404];

/**
 * Standard input sizes for different detectors
 */
export const INPUT_SIZES = {
  SSD_MOBILENET_V1: 512,
  TINY_FACE_DETECTOR: 416,
  TINY_YOLOV2: 416,
} as const;

/**
 * Default IOU (Intersection over Union) thresholds
 */
export const IOU_THRESHOLDS = {
  DEFAULT: 0.5,
  TINY_FACE_DETECTOR: 0.45,
  TINY_YOLOV2: 0.5,
} as const;

/**
 * Number of facial landmarks
 */
export const LANDMARKS = {
  FULL: 68,
  TINY: 5,
  COORDS_PER_POINT: 2,
} as const;

/**
 * Face expression labels in order
 */
export const EXPRESSION_LABELS = [
  'neutral',
  'happy',
  'sad',
  'angry',
  'fearful',
  'disgusted',
  'surprised',
] as const;

export type ExpressionLabel = typeof EXPRESSION_LABELS[number];

/**
 * Face recognition descriptor size
 */
export const FACE_DESCRIPTOR_LENGTH = 128;

/**
 * Mapping from FaceMesh 468 landmarks to traditional 68-point landmarks.
 *
 * The 68-point format is:
 * - 0-16: Jaw line (17 points)
 * - 17-21: Left eyebrow (5 points)
 * - 22-26: Right eyebrow (5 points)
 * - 27-35: Nose (9 points)
 * - 36-41: Left eye (6 points)
 * - 42-47: Right eye (6 points)
 * - 48-67: Mouth (20 points)
 */

// Jaw line (17 points)
const JAW_LINE = [
  10, 338, 297, 332, 284, 251, 389, 356, 454,
  323, 361, 288, 397, 365, 379, 378, 152,
];

// Left eyebrow (5 points)
const LEFT_EYEBROW = [70, 63, 105, 66, 107];

// Right eyebrow (5 points)
const RIGHT_EYEBROW = [336, 296, 334, 293, 300];

// Nose (9 points)
const NOSE = [
  168, // Bridge top
  6, // Bridge
  197, // Bridge bottom
  195, // Tip
  5, // Bottom center
  4, // Left nostril
  1, // Tip
  2, // Right nostril
  98, // Left side
];

// Left eye (6 points)
const LEFT_EYE = [33, 160, 158, 133, 153, 144];

// Right eye (6 points)
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

// Mouth outer (12 points)
const MOUTH_OUTER = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
];

// Mouth inner (8 points)
const MOUTH_INNER = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324,
].slice(0, 8);

/**
 * Complete mapping from FaceMesh indices to 68-point format.
 * Each index in this array corresponds to a 68-point landmark,
 * and the value is the corresponding FaceMesh index.
 */
export const FACEMESH_TO_68_MAPPING: number[] = [
  // Jaw line (0-16)
  ...JAW_LINE,

  // Left eyebrow (17-21)
  ...LEFT_EYEBROW,

  // Right eyebrow (22-26)
  ...RIGHT_EYEBROW,

  // Nose (27-35)
  ...NOSE,

  // Left eye (36-41)
  ...LEFT_EYE,

  // Right eye (42-47)
  ...RIGHT_EYE,

  // Mouth outer (48-59)
  ...MOUTH_OUTER,

  // Mouth inner (60-67)
  ...MOUTH_INNER,
];

/**
 * Semantic region indices for FaceMesh.
 */
export const FACEMESH_REGIONS = {
  FACE_OVAL: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  ],

  LEFT_EYE: [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  ],

  RIGHT_EYE: [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
  ],

  LEFT_EYEBROW: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],

  RIGHT_EYEBROW: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],

  NOSE: [
    1, 2, 98, 327, 4, 5, 6, 168, 195, 197, 419, 351, 412, 343,
    437, 420, 456, 248, 281, 275, 274, 354, 370, 94, 19,
  ],

  LIPS_OUTER: [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
  ],

  LIPS_INNER: [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
  ],

  // Iris landmarks (only available with refinement)
  LEFT_IRIS: [468, 469, 470, 471, 472],
  RIGHT_IRIS: [473, 474, 475, 476, 477],
} as const;

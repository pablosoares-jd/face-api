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
/**
 * Complete mapping from FaceMesh indices to 68-point format.
 * Each index in this array corresponds to a 68-point landmark,
 * and the value is the corresponding FaceMesh index.
 */
export declare const FACEMESH_TO_68_MAPPING: number[];
/**
 * Semantic region indices for FaceMesh.
 */
export declare const FACEMESH_REGIONS: {
    readonly FACE_OVAL: readonly [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
    readonly LEFT_EYE: readonly [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    readonly RIGHT_EYE: readonly [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
    readonly LEFT_EYEBROW: readonly [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
    readonly RIGHT_EYEBROW: readonly [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
    readonly NOSE: readonly [1, 2, 98, 327, 4, 5, 6, 168, 195, 197, 419, 351, 412, 343, 437, 420, 456, 248, 281, 275, 274, 354, 370, 94, 19];
    readonly LIPS_OUTER: readonly [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    readonly LIPS_INNER: readonly [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
    readonly LEFT_IRIS: readonly [468, 469, 470, 471, 472];
    readonly RIGHT_IRIS: readonly [473, 474, 475, 476, 477];
};

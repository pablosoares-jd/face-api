import { describe, it, expect } from 'vitest';
import { euclideanDistance } from '../euclideanDistance';

describe('euclideanDistance', () => {
  describe('basic calculations', () => {
    it('should return 0 for identical arrays', () => {
      const arr = [1, 2, 3, 4, 5];
      expect(euclideanDistance(arr, arr)).toBe(0);
    });

    it('should calculate distance between two simple arrays', () => {
      const arr1 = [0, 0];
      const arr2 = [3, 4];
      expect(euclideanDistance(arr1, arr2)).toBe(5);
    });

    it('should calculate distance for 3D vectors', () => {
      const arr1 = [1, 2, 3];
      const arr2 = [4, 6, 3];
      expect(euclideanDistance(arr1, arr2)).toBe(5);
    });

    it('should handle negative values', () => {
      const arr1 = [-1, -2];
      const arr2 = [2, 2];
      expect(euclideanDistance(arr1, arr2)).toBe(5);
    });

    it('should handle floating point numbers', () => {
      const arr1 = [0.5, 0.5];
      const arr2 = [1.5, 1.5];
      expect(euclideanDistance(arr1, arr2)).toBeCloseTo(Math.sqrt(2));
    });
  });

  describe('Float32Array support', () => {
    it('should work with Float32Array inputs', () => {
      const arr1 = new Float32Array([0, 0]);
      const arr2 = new Float32Array([3, 4]);
      expect(euclideanDistance(arr1, arr2)).toBe(5);
    });

    it('should work with mixed array types', () => {
      const arr1 = [0, 0];
      const arr2 = new Float32Array([3, 4]);
      expect(euclideanDistance(arr1, arr2)).toBe(5);
    });
  });

  describe('error handling', () => {
    it('should throw error for arrays of different lengths', () => {
      const arr1 = [1, 2, 3];
      const arr2 = [1, 2];
      expect(() => euclideanDistance(arr1, arr2)).toThrow('euclideanDistance: arr1.length !== arr2.length');
    });

    it('should handle empty arrays', () => {
      const arr1: number[] = [];
      const arr2: number[] = [];
      expect(euclideanDistance(arr1, arr2)).toBe(0);
    });
  });

  describe('face descriptor scenarios', () => {
    it('should calculate distance for typical face descriptor length (128)', () => {
      const arr1 = new Array(128).fill(0.1);
      const arr2 = new Array(128).fill(0.2);
      const result = euclideanDistance(arr1, arr2);
      expect(result).toBeCloseTo(Math.sqrt(128 * 0.01));
    });

    it('should return small distance for similar descriptors', () => {
      const arr1 = new Array(128).fill(0.5);
      const arr2 = arr1.map((v, i) => v + (i % 2 === 0 ? 0.001 : -0.001));
      const result = euclideanDistance(arr1, arr2);
      expect(result).toBeLessThan(0.1);
    });

    it('should return larger distance for different descriptors', () => {
      const arr1 = new Array(128).fill(0.1);
      const arr2 = new Array(128).fill(0.9);
      const result = euclideanDistance(arr1, arr2);
      expect(result).toBeGreaterThan(1);
    });
  });
});

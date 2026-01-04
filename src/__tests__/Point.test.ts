import { describe, it, expect } from 'vitest';
import { Point } from '../classes/Point';

describe('Point', () => {
  describe('constructor and getters', () => {
    it('should create a point with correct coordinates', () => {
      const point = new Point(10, 20);
      expect(point.x).toBe(10);
      expect(point.y).toBe(20);
    });

    it('should handle negative coordinates', () => {
      const point = new Point(-5, -10);
      expect(point.x).toBe(-5);
      expect(point.y).toBe(-10);
    });

    it('should handle zero coordinates', () => {
      const point = new Point(0, 0);
      expect(point.x).toBe(0);
      expect(point.y).toBe(0);
    });

    it('should handle floating point coordinates', () => {
      const point = new Point(1.5, 2.7);
      expect(point.x).toBeCloseTo(1.5);
      expect(point.y).toBeCloseTo(2.7);
    });
  });

  describe('add', () => {
    it('should add two points correctly', () => {
      const p1 = new Point(3, 4);
      const p2 = new Point(1, 2);
      const result = p1.add(p2);
      expect(result.x).toBe(4);
      expect(result.y).toBe(6);
    });

    it('should handle negative values in addition', () => {
      const p1 = new Point(5, 5);
      const p2 = new Point(-3, -2);
      const result = p1.add(p2);
      expect(result.x).toBe(2);
      expect(result.y).toBe(3);
    });

    it('should return a new Point instance', () => {
      const p1 = new Point(1, 1);
      const p2 = new Point(2, 2);
      const result = p1.add(p2);
      expect(result).not.toBe(p1);
      expect(result).not.toBe(p2);
    });
  });

  describe('sub', () => {
    it('should subtract two points correctly', () => {
      const p1 = new Point(10, 8);
      const p2 = new Point(3, 2);
      const result = p1.sub(p2);
      expect(result.x).toBe(7);
      expect(result.y).toBe(6);
    });

    it('should handle resulting negative values', () => {
      const p1 = new Point(2, 3);
      const p2 = new Point(5, 7);
      const result = p1.sub(p2);
      expect(result.x).toBe(-3);
      expect(result.y).toBe(-4);
    });
  });

  describe('mul', () => {
    it('should multiply two points correctly', () => {
      const p1 = new Point(3, 4);
      const p2 = new Point(2, 3);
      const result = p1.mul(p2);
      expect(result.x).toBe(6);
      expect(result.y).toBe(12);
    });

    it('should handle zero multiplication', () => {
      const p1 = new Point(5, 10);
      const p2 = new Point(0, 0);
      const result = p1.mul(p2);
      expect(result.x).toBe(0);
      expect(result.y).toBe(0);
    });
  });

  describe('div', () => {
    it('should divide two points correctly', () => {
      const p1 = new Point(10, 12);
      const p2 = new Point(2, 3);
      const result = p1.div(p2);
      expect(result.x).toBe(5);
      expect(result.y).toBe(4);
    });

    it('should handle division resulting in floating point', () => {
      const p1 = new Point(5, 7);
      const p2 = new Point(2, 3);
      const result = p1.div(p2);
      expect(result.x).toBe(2.5);
      expect(result.y).toBeCloseTo(7 / 3);
    });
  });

  describe('abs', () => {
    it('should return absolute values', () => {
      const point = new Point(-5, -10);
      const result = point.abs();
      expect(result.x).toBe(5);
      expect(result.y).toBe(10);
    });

    it('should keep positive values unchanged', () => {
      const point = new Point(3, 7);
      const result = point.abs();
      expect(result.x).toBe(3);
      expect(result.y).toBe(7);
    });
  });

  describe('magnitude', () => {
    it('should calculate magnitude correctly for 3-4-5 triangle', () => {
      const point = new Point(3, 4);
      expect(point.magnitude()).toBe(5);
    });

    it('should return 0 for origin point', () => {
      const point = new Point(0, 0);
      expect(point.magnitude()).toBe(0);
    });

    it('should handle unit vector', () => {
      const point = new Point(1, 0);
      expect(point.magnitude()).toBe(1);
    });

    it('should calculate magnitude for negative values', () => {
      const point = new Point(-3, -4);
      expect(point.magnitude()).toBe(5);
    });
  });

  describe('floor', () => {
    it('should floor coordinates correctly', () => {
      const point = new Point(3.7, 4.9);
      const result = point.floor();
      expect(result.x).toBe(3);
      expect(result.y).toBe(4);
    });

    it('should handle negative floating point values', () => {
      const point = new Point(-2.3, -1.8);
      const result = point.floor();
      expect(result.x).toBe(-3);
      expect(result.y).toBe(-2);
    });

    it('should not change integer values', () => {
      const point = new Point(5, 10);
      const result = point.floor();
      expect(result.x).toBe(5);
      expect(result.y).toBe(10);
    });
  });
});

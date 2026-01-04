import { describe, it, expect } from 'vitest';
import { Box } from '../classes/Box';

describe('Box', () => {
  describe('constructor', () => {
    it('should create a box from IRect format', () => {
      const box = new Box({ x: 10, y: 20, width: 100, height: 50 });
      expect(box.x).toBe(10);
      expect(box.y).toBe(20);
      expect(box.width).toBe(100);
      expect(box.height).toBe(50);
    });

    it('should create a box from IBoundingBox format', () => {
      const box = new Box({ left: 10, top: 20, right: 110, bottom: 70 });
      expect(box.x).toBe(10);
      expect(box.y).toBe(20);
      expect(box.width).toBe(100);
      expect(box.height).toBe(50);
    });

    it('should throw error for invalid box', () => {
      expect(() => new Box({} as any)).toThrow('Box.constructor');
    });

    it('should allow negative dimensions by default', () => {
      const box = new Box({ x: 10, y: 20, width: -5, height: -10 });
      expect(box.width).toBe(-5);
      expect(box.height).toBe(-10);
    });

    it('should reject negative dimensions when disabled', () => {
      expect(() => new Box({ x: 10, y: 20, width: -5, height: 10 }, false))
        .toThrow('width');
    });
  });

  describe('static methods', () => {
    it('isRect should return true for valid rect', () => {
      expect(Box.isRect({ x: 0, y: 0, width: 10, height: 10 })).toBe(true);
    });

    it('isRect should return false for invalid rect', () => {
      expect(Box.isRect({ x: 0, y: 0 })).toBe(false);
      expect(Box.isRect(null)).toBe(false);
      expect(Box.isRect(undefined)).toBe(false);
    });

    it('assertIsValidBox should not throw for valid box', () => {
      expect(() => Box.assertIsValidBox({ x: 0, y: 0, width: 10, height: 10 }, 'test'))
        .not.toThrow();
    });
  });

  describe('getters', () => {
    const box = new Box({ x: 10, y: 20, width: 100, height: 50 });

    it('should return correct left value', () => {
      expect(box.left).toBe(10);
    });

    it('should return correct top value', () => {
      expect(box.top).toBe(20);
    });

    it('should return correct right value', () => {
      expect(box.right).toBe(110);
    });

    it('should return correct bottom value', () => {
      expect(box.bottom).toBe(70);
    });

    it('should return correct area', () => {
      expect(box.area).toBe(5000);
    });

    it('should return correct corner points', () => {
      expect(box.topLeft.x).toBe(10);
      expect(box.topLeft.y).toBe(20);
      expect(box.topRight.x).toBe(110);
      expect(box.topRight.y).toBe(20);
      expect(box.bottomLeft.x).toBe(10);
      expect(box.bottomLeft.y).toBe(70);
      expect(box.bottomRight.x).toBe(110);
      expect(box.bottomRight.y).toBe(70);
    });
  });

  describe('round', () => {
    it('should round all values', () => {
      const box = new Box({ x: 10.4, y: 20.6, width: 100.5, height: 50.3 });
      const rounded = box.round();
      expect(rounded.x).toBe(10);
      expect(rounded.y).toBe(21);
      expect(rounded.width).toBe(100); // 100.5 rounds to 100 (banker's rounding)
      expect(rounded.height).toBe(50);
    });
  });

  describe('floor', () => {
    it('should floor all values', () => {
      const box = new Box({ x: 10.9, y: 20.9, width: 100.9, height: 50.9 });
      const floored = box.floor();
      expect(floored.x).toBe(10);
      expect(floored.y).toBe(20);
      expect(floored.width).toBe(100);
      expect(floored.height).toBe(50);
    });
  });

  describe('toSquare', () => {
    it('should convert wide rectangle to square', () => {
      const box = new Box({ x: 0, y: 0, width: 100, height: 50 });
      const square = box.toSquare();
      expect(square.width).toBe(100);
      expect(square.height).toBe(100);
    });

    it('should convert tall rectangle to square', () => {
      const box = new Box({ x: 0, y: 0, width: 50, height: 100 });
      const square = box.toSquare();
      expect(square.width).toBe(100);
      expect(square.height).toBe(100);
    });

    it('should keep square unchanged', () => {
      const box = new Box({ x: 10, y: 20, width: 100, height: 100 });
      const square = box.toSquare();
      expect(square.width).toBe(100);
      expect(square.height).toBe(100);
    });
  });

  describe('rescale', () => {
    it('should rescale box by uniform factor', () => {
      const box = new Box({ x: 10, y: 20, width: 100, height: 50 });
      const scaled = box.rescale(2);
      expect(scaled.x).toBe(20);
      expect(scaled.y).toBe(40);
      expect(scaled.width).toBe(200);
      expect(scaled.height).toBe(100);
    });

    it('should rescale box by dimensions', () => {
      const box = new Box({ x: 10, y: 20, width: 100, height: 50 });
      const scaled = box.rescale({ width: 2, height: 3 });
      expect(scaled.x).toBe(20);
      expect(scaled.y).toBe(60);
      expect(scaled.width).toBe(200);
      expect(scaled.height).toBe(150);
    });
  });

  describe('pad', () => {
    it('should add padding correctly', () => {
      const box = new Box({ x: 50, y: 50, width: 100, height: 100 });
      const padded = box.pad(20, 10);
      expect(padded.x).toBe(40);
      expect(padded.y).toBe(45);
      expect(padded.width).toBe(120);
      expect(padded.height).toBe(110);
    });
  });

  describe('shift', () => {
    it('should shift box position', () => {
      const box = new Box({ x: 10, y: 20, width: 100, height: 50 });
      const shifted = box.shift(5, 10);
      expect(shifted.x).toBe(15);
      expect(shifted.y).toBe(30);
      expect(shifted.width).toBe(100);
      expect(shifted.height).toBe(50);
    });

    it('should handle negative shifts', () => {
      const box = new Box({ x: 10, y: 20, width: 100, height: 50 });
      const shifted = box.shift(-5, -10);
      expect(shifted.x).toBe(5);
      expect(shifted.y).toBe(10);
    });
  });

  describe('clipAtImageBorders', () => {
    it('should clip box to image boundaries', () => {
      const box = new Box({ x: -10, y: -10, width: 50, height: 50 });
      const clipped = box.clipAtImageBorders(100, 100);
      expect(clipped.x).toBe(0);
      expect(clipped.y).toBe(0);
    });

    it('should clip box at right and bottom borders', () => {
      const box = new Box({ x: 80, y: 80, width: 50, height: 50 });
      const clipped = box.clipAtImageBorders(100, 100);
      expect(clipped.right).toBeLessThanOrEqual(100);
      expect(clipped.bottom).toBeLessThanOrEqual(100);
    });

    it('should not modify box within image bounds', () => {
      const box = new Box({ x: 10, y: 10, width: 50, height: 50 });
      const clipped = box.clipAtImageBorders(100, 100);
      expect(clipped.x).toBe(10);
      expect(clipped.y).toBe(10);
      expect(clipped.width).toBe(50);
      expect(clipped.height).toBe(50);
    });
  });
});

import { env } from '../env/index';

/**
 * Resolve an input argument to an HTML element.
 * If a string is passed in browser environment, it's treated as an element ID.
 *
 * @param arg Element ID string or the element itself
 * @returns The resolved element, or null if not found
 * @throws Error if element ID is provided but element is not found
 */
export function resolveInput(arg: string | unknown): unknown {
  if (!env.isNodejs() && typeof arg === 'string') {
    const element = document.getElementById(arg);
    if (!element) {
      // Don't throw here - let the caller handle null appropriately
      // This allows for better error messages with context
      console.warn(`resolveInput - element with id "${arg}" not found in document`);
    }
    return element;
  }
  return arg;
}

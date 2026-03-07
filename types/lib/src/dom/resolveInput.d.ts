/**
 * Resolve an input argument to an HTML element.
 * If a string is passed in browser environment, it's treated as an element ID.
 *
 * @param arg Element ID string or the element itself
 * @returns The resolved element, or null if not found
 * @throws Error if element ID is provided but element is not found
 */
export declare function resolveInput(arg: string | unknown): unknown;

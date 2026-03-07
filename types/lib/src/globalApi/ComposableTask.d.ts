/**
 * Base class for composable async tasks in face-api.
 * Implements a Promise-like interface for chaining operations.
 */
export declare class ComposableTask<T> {
    /**
     * Implements the Promise thenable interface.
     * @param onfulfilled Called when the task completes successfully
     * @param onrejected Called when the task fails (optional)
     * @returns Promise that resolves with the transformed value
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null, onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null): Promise<TResult1 | TResult2>;
    /**
     * Implements the Promise catch interface.
     * @param onrejected Called when the task fails
     * @returns Promise that handles the rejection
     */
    catch<TResult = never>(onrejected?: ((reason: unknown) => TResult | PromiseLike<TResult>) | null): Promise<T | TResult>;
    /**
     * Execute the task. Must be implemented by subclasses.
     * @returns Promise with the task result
     */
    run(): Promise<T>;
}

from U_support import REPLwithU
import numpy as np
from typing import Tuple, List, Optional
import sys

np.set_printoptions(precision=8, suppress=True, linewidth=120)


class AttentionTester:    
    def __init__(self, matrix_size: int = 3):
        self.matrix_size = matrix_size
        self.total_elements = matrix_size * matrix_size
    
    def get_user_input(self) -> Tuple[List[float], List[float], List[float]]:
        print(f"\nEnter {self.total_elements} values for each matrix (space-separated):")
        print(f"These will be reshaped into {self.matrix_size}x{self.matrix_size} matrices.\n")
        
        def get_matrix(name: str) -> List[float]:
            while True:
                try:
                    user_input = input(f"Matrix {name}: ").strip()
                    values = [float(x) for x in user_input.split()]
                    if len(values) != self.total_elements:
                        print(f"Error: Expected {self.total_elements} values, got {len(values)}")
                        continue
                    return values
                except ValueError:
                    print("Error: Please enter valid numbers separated by spaces")
                except KeyboardInterrupt:
                    print("\nInput cancelled.")
                    sys.exit(0)
        
        propA = get_matrix("A (attention weights)")
        propX = get_matrix("X (input)")
        propV = get_matrix("V (value)")
        
        return propA, propX, propV
    
    def generate_random_input(self, seed: Optional[int] = None) -> Tuple[List[float], List[float], List[float]]:
        if seed is not None:
            np.random.seed(seed)
        
        propA = np.random.uniform(0.1, 1.0, self.total_elements).tolist()
        propX = np.random.uniform(0.1, 1.0, self.total_elements).tolist()
        propV = np.random.uniform(0.1, 1.0, self.total_elements).tolist()
        
        return propA, propX, propV
    
    def run_rasp_implementation(self, A: List[float], X: List[float], V: List[float]) -> Tuple:
        executor = REPLwithU()
        executor.env.set_variable('X', X)
        executor.run_given_line('set example X')
        executor.run_given_line("examples off")
        
        # Step 1: Transpose X
        executor.run_given_line('XT = Transpose_3dot()(X);')
        X_T = executor.env.get_variable('XT').val._vals
        
        # Step 2: Y1 = X @ A
        executor.env.set_variable('XA', X + A)
        executor.run_given_line('Y1 = matmul_3dot3()(XA);')
        Y1 = executor.env.get_variable('Y1').val._vals[:self.total_elements]
        
        # Step 3: Y2 = Y1 @ X^T
        executor.env.set_variable('Y1XT', Y1 + X_T)
        executor.run_given_line('Y2 = matmul_3dot3()(Y1XT);')
        Y2 = executor.env.get_variable('Y2').val._vals[:self.total_elements]
        
        # Step 4: Y3 = softmax(Y2)
        executor.env.set_variable('Y2', Y2)
        executor.run_given_line('Y3 = softmax_3dot()(Y2);')
        Y3 = executor.env.get_variable('Y3').val._vals
        
        # Step 5: Y4 = X @ V
        executor.env.set_variable('XV', X + V)
        executor.run_given_line('Y4 = matmul_3dot3()(XV);')
        Y4 = executor.env.get_variable('Y4').val._vals[:self.total_elements]
        
        # Step 6: Y = Y3 @ Y4
        executor.env.set_variable('Y3Y4', Y3 + Y4)
        executor.run_given_line('Y = matmul_3dot3()(Y3Y4);')
        Y = executor.env.get_variable('Y').val._vals[:self.total_elements]
        
        return X_T, Y1, Y2, Y3, Y4, Y
    
    @staticmethod
    def softmax_rows(mat: np.ndarray) -> np.ndarray:
        m = np.asarray(mat, dtype=float)
        row_max = np.max(m, axis=1, keepdims=True)
        e = np.exp(m - row_max)
        return e / np.sum(e, axis=1, keepdims=True)
    
    def run_numpy_implementation(self, A_flat: List[float], X_flat: List[float], 
                                 V_flat: List[float]) -> Tuple:
        n = self.matrix_size
        A = np.asarray(A_flat, dtype=float).reshape((n, n))
        X = np.asarray(X_flat, dtype=float).reshape((n, n))
        V = np.asarray(V_flat, dtype=float).reshape((n, n))
        
        # Attention computation: Y = softmax(X @ A @ X^T) @ (X @ V)
        X_T = X.T
        Y1 = X @ A
        Y2 = Y1 @ X_T
        Y3 = self.softmax_rows(Y2)
        Y4 = X @ V
        Y = Y3 @ Y4
        
        return X_T, Y1, Y2, Y3, Y4, Y
    
    def pretty_print_results(self, rasp_results: Tuple, numpy_results: Tuple):
        labels = ["X^T", "Y1 (X·A)", "Y2 (Y1·X^T)", "Y3 (softmax)", "Y4 (X·V)", "Y (final)"]
        n = self.matrix_size
        
        print("\n" + "="*80)
        print("RASP RESULTS")
        print("="*80)
        for label, result in zip(labels, rasp_results):
            print(f"\n{label}:")
            result_array = np.asarray(result, dtype=float).reshape(-1) #.reshape(n, n)
            print(result_array)
        
        print("\n" + "="*80)
        print("NUMPY RESULTS")
        print("="*80)
        for label, result in zip(labels, numpy_results):
            print(f"\n{label}:")
            result_array = np.asarray(result, dtype=float).reshape(-1)  #.reshape(n, n)
            print(result_array)
        
        print("\n" + "="*80)
        print("ABSOLUTE DIFFERENCES")
        print("="*80)
        for label, rasp_res, numpy_res in zip(labels, rasp_results, numpy_results):
            print(f"\n{label}:")
            rasp_array = np.asarray(rasp_res, dtype=float).reshape(n, n)
            numpy_array = np.asarray(numpy_res, dtype=float).reshape(n, n)
            abs_diff = np.abs(rasp_array - numpy_array)
            print(abs_diff.reshape(-1))
    
    @staticmethod
    def compute_difference(x: np.ndarray, y: np.ndarray, 
                          tolerance: float = 1e-3) -> Tuple[np.ndarray, dict]:
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        abs_diff = np.abs(x - y)
        stats = {
            'max_diff': float(np.max(abs_diff)),
            'mean_diff': float(np.mean(abs_diff)),
            'std_diff': float(np.std(abs_diff)),
            'is_close': np.allclose(x, y, rtol=tolerance, atol=tolerance)
        }
        
        if not stats['is_close']:
            raise AssertionError(
                f"Results differ significantly!\n"
                f"Max difference: {stats['max_diff']:.6e}\n"
                f"Mean difference: {stats['mean_diff']:.6e}"
            )
        
        return abs_diff, stats
    
    def run_comparison(self, propA: List[float], propX: List[float], 
                      propV: List[float], verbose: bool = True):
        print("\n" + "="*80)
        print("Running RASP Implementation...")
        rasp_results = self.run_rasp_implementation(propA, propX, propV)
        
        print("\nRunning NumPy Implementation...")
        numpy_results = self.run_numpy_implementation(propA, propX, propV)
        
        if verbose:
            self.pretty_print_results(rasp_results, numpy_results)
        
        # Validation
        rasp_arrays = [np.asarray(r, dtype=float).flatten() for r in rasp_results]
        numpy_arrays = [r.flatten() for r in numpy_results]
        
        print("\n" + "="*80)
        print("VALIDATION")
        print("="*80)
        
        labels = ["X^T", "Y1 (X·A)", "Y2 (Y1·X^T)", "Y3 (softmax)", "Y4 (X·V)", "Y (final)"]
        all_close = True
        
        for label, rasp, numpy in zip(labels, rasp_arrays, numpy_arrays):
            try:
                diff, stats = self.compute_difference(rasp, numpy)
                status = "✓ PASS" if stats['is_close'] else "✗ FAIL"
                print(f"{label:15s} {status}  (max diff: {stats['max_diff']:.6e})")
            except AssertionError as e:
                print(f"{label:15s} ✗ FAIL")
                all_close = False
        
        print("="*80)
        if all_close:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ TESTS FAILED")
        print("="*80 + "\n")
        
        return rasp_results, numpy_results


def main():
    tester = AttentionTester(matrix_size=3)
    
    print("="*80)
    print("ATTENTION MECHANISM TESTER")
    print("="*80)
    print("\nOptions:")
    print("  1. Enter custom values")
    print("  2. Generate random values")
    print("  3. Use default test values")
    
    while True:
        try:
            choice = input("\nSelect option (1/2/3): ").strip()
            
            if choice == "1":
                propA, propX, propV = tester.get_user_input()
                break
            elif choice == "2":
                seed_input = input("Enter random seed (or press Enter for random): ").strip()
                seed = int(seed_input) if seed_input else None
                propA, propX, propV = tester.generate_random_input(seed)
                print(f"\nGenerated random values (seed={seed})")
                break
            elif choice == "3":
                propA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                propV = [0.1, 0.7, 0.4, 1.0, 0.3, 0.9, 0.5, 0.2, 0.6]
                propX = [0.8, 0.20, 0.5, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7]
                print("\nUsing default test values")
                break
            else:
                print("Invalid option. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except ValueError:
            print("Invalid input. Please try again.")
    
    rasp_results, numpy_results = tester.run_comparison(propA, propX, propV, verbose=True)


if __name__ == "__main__":
    main()

// Solving a system, and checking the answer by residual rather than by eye.
// Run: bantu run samples/numba/03_linear_algebra.b
include "../../numba/numba.b" as np;

np.seed(11);
$n = 200;
// Diagonally dominant, so it is certainly non-singular.
$A = np.add(np.randn([$n, $n], null, null), np.multiply(np.identity($n), $n * 1.0, null), null);
$b = np.randn([$n], null, null);

$t0 = clock();
$x = np.solve($A, $b);
print("solved " + str($n) + "x" + str($n) + " in " + str(clock() - $t0) + " ms");

$resid = np.get(np.max(np.abs(np.subtract(np.matmul($A, $x), $b, null), null), null, null), []);
print("residual ||Ax - b||_inf = " + str($resid));
print("determinant sign/log    = " + str(np.to_list(np.slogdet($A))));
print("condition number        = " + str(np.get(np.cond(np.identity(4)), [])) + " for the identity");

// A symmetric matrix: eigenvalues and a reconstruction check.
$S = np.matmul($A, np.T($A));
$e = np.eigh($S);
print("smallest eigenvalue     = " + str(np.get($e[0], [0])));
print("eigenvectors orthonormal to " +
      str(np.get(np.max(np.abs(np.subtract(np.matmul(np.T($e[1]), $e[1]), np.identity($n), null), null), null, null), [])));

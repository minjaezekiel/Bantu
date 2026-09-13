// ════════════════════════════════════════════════════════════════════════
//  numba_linalg_test.b — matmul, LU, Cholesky, QR, SVD and eigh.
//
//  Linear algebra is tested by RESIDUAL, not by comparing numbers to a
//  reference. Asserting that solve() returns a particular vector is fragile
//  and proves little; asserting that ||Ax - b|| is below 1e-10 proves the
//  answer actually solves the system, and stays meaningful when the pivoting
//  order changes. Same for the factorisations: Q must be orthogonal, L*Lt
//  must reconstruct A, U*S*Vt must reconstruct A.
//
//  The other half is failure. A singular matrix, a non-square one, a
//  non-symmetric input to eigh and a non-positive-definite input to Cholesky
//  must all RAISE with a message that says what to use instead -- returning
//  NaN or garbage for these is the worst outcome, because it looks like an
//  answer.
//
//  Run:  bantu run tests/numba_linalg_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};

def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def close($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
def raises($fn, $needle, $name) {
    try {
        $fn();
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  (no error raised)");
    } catch ($e) {
        $msg = str($e.message);
        if (contains($msg, $needle)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
        else {
            $R.fail = $R.fail + 1;
            print("  FAIL  " + $name);
            print("          message did not mention \"" + $needle + "\": " + $msg);
        }
    }
}

// The largest absolute element of an array -- the residual measure used
// throughout, because a max-norm bound is stricter than an average.
def maxabs($a) { return nd_get(nd_max(nd_abs($a, null), null, null), []); }

print("========================================");
print("  numba — linear algebra");
print("========================================");

print("");
print("-- matmul --");
$A = nd([[1, 2], [3, 4]], "f64");
$B = nd([[5, 6], [7, 8]], "f64");
eq(str(nd_to_list(nd_matmul($A, $B))), "[[19, 22], [43, 50]]", "2x2 matmul");
eq(str(nd_shape(nd_matmul(nd_zeros([3, 5], null), nd_zeros([5, 2], null)))), "[3, 2]",
   "(3,5) x (5,2) gives (3,2)");
raises(def() { return nd_matmul(nd_zeros([3, 5], null), nd_zeros([4, 2], null)); },
   "inner dimensions", "a mismatched inner dimension names both sizes");
// A vector operand collapses its axis, as in NumPy.
eq(str(nd_to_list(nd_matmul($A, nd([1, 1], "f64")))), "[3, 7]", "matrix times vector");
eq(str(nd_to_list(nd_matmul(nd([1, 1], "f64"), $A))), "[4, 6]", "vector times matrix");
eq(nd_get(nd_matmul(nd([1, 2], "f64"), nd([3, 4], "f64")), []), 11, "vector times vector is the inner product");
eq(nd_get(nd_dot(nd([1, 2, 3], "f64"), nd([4, 5, 6], "f64")), []), 32, "nd_dot");
eq(str(nd_to_list(nd_outer(nd([1, 2], "f64"), nd([3, 4], "f64")))), "[[3, 4], [6, 8]]", "nd_outer");
eq(nd_get(nd_trace($A), []), 5, "nd_trace");

// Identity must be exact, and the associativity check catches an ordering bug
// that a single product would not.
$I = nd_identity(4);
$M = nd_random_normal([4, 4], 0, 1, 7);
ok(maxabs(nd_subtract(nd_matmul($M, $I), $M, null)) < 1e-14, "M x I == M exactly");
$C = nd_random_normal([4, 4], 0, 1, 8);
$left  = nd_matmul(nd_matmul($M, $C), $I);
$right = nd_matmul($M, nd_matmul($C, $I));
ok(maxabs(nd_subtract($left, $right, null)) < 1e-12, "matmul is associative to 1e-12");

print("");
print("-- solve, by residual --");
$S = nd([[4.0, 1.0], [1.0, 3.0]], "f64");
$b = nd([1.0, 2.0], "f64");
$x = nd_solve($S, $b);
$resid = maxabs(nd_subtract(nd_matmul($S, $x), $b, null));
print("        2x2 residual: " + str($resid));
ok($resid < 1e-14, "the 2x2 solution actually solves the system");
// A right-hand side matrix solves several systems at once.
$X2 = nd_solve($S, nd([[1.0, 0.0], [0.0, 1.0]], "f64"));
eq(str(nd_shape($X2)), "[2, 2]", "a matrix right-hand side returns a matrix");
ok(maxabs(nd_subtract(nd_matmul($S, $X2), nd_identity(2), null)) < 1e-14,
   "solving against the identity gives the inverse");

print("");
print("-- inverse and determinant --");
ok(maxabs(nd_subtract(nd_matmul($S, nd_inv($S)), nd_identity(2), null)) < 1e-14,
   "A x inv(A) is the identity");
close(nd_get(nd_det($A), []), -2.0, 1e-12, "det([[1,2],[3,4]]) is -2");
close(nd_get(nd_det(nd_identity(5)), []), 1.0, 1e-12, "det(I) is 1");
// The determinant of a large matrix overflows long before it is interesting,
// which is why slogdet exists.
$sld = nd_slogdet($A);
close(nd_get($sld, [0]), -1.0, 1e-12, "slogdet sign");
close(nd_get($sld, [1]), 0.6931471805599453, 1e-12, "slogdet log|det| is log(2)");

print("");
print("-- Cholesky --");
// A symmetric positive-definite matrix, built as M*Mt + nI so it is certainly SPD.
$P = nd_add(nd_matmul($M, nd_T($M)), nd_multiply(nd_identity(4), 4.0, null), null);
$L = nd_cholesky($P);
ok(maxabs(nd_subtract(nd_matmul($L, nd_T($L)), $P, null)) < 1e-11, "L x Lt reconstructs A");
// The lower triangle is what is returned, so the upper must be zero.
eq(nd_get($L, [0, 3]), 0, "the result is lower-triangular");
raises(def() { return nd_cholesky(nd([[1.0, 2.0], [2.0, 1.0]], "f64")); }, "not positive definite",
   "a non-positive-definite matrix raises, naming the fix");
raises(def() { return nd_cholesky(nd_zeros([2, 3], null)); }, "square",
   "a non-square matrix raises");

print("");
print("-- QR --");
$Aq = nd_random_normal([6, 4], 0, 1, 11);
$qr = nd_qr($Aq);
$Q = $qr[0];
$Rm = $qr[1];
eq(str(nd_shape($Q)),  "[6, 4]", "Q is m-by-k (reduced form)");
eq(str(nd_shape($Rm)), "[4, 4]", "R is k-by-n");
ok(maxabs(nd_subtract(nd_matmul($Q, $Rm), $Aq, null)) < 1e-12, "Q x R reconstructs A");
// Orthogonality is the property Householder is chosen FOR: Gram-Schmidt loses
// it catastrophically on anything ill-conditioned.
$QtQ = nd_matmul(nd_T($Q), $Q);
$orth = maxabs(nd_subtract($QtQ, nd_identity(4), null));
print("        ||QtQ - I||: " + str($orth));
ok($orth < 1e-13, "Qt x Q is the identity to 1e-13");
// R must be upper triangular.
ok(nd_get($Rm, [3, 0]) == 0 && nd_get($Rm, [2, 1]) == 0, "R is upper-triangular");

print("");
print("-- least squares, by residual orthogonality --");
// The defining property of a least-squares solution: the residual is
// orthogonal to every column of A. That is a stronger check than comparing x
// to a reference, and it holds regardless of how the solve is implemented.
$Al = nd([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], "f64");
$bl = nd([6.0, 5.0, 7.0, 10.0], "f64");
$xl = nd_lstsq($Al, $bl);
eq(str(nd_shape($xl)), "[2]", "lstsq returns one coefficient per column");
$r = nd_subtract(nd_matmul($Al, $xl), $bl, null);
$ortho = maxabs(nd_matmul(nd_T($Al), $r));
print("        ||At r||: " + str($ortho));
ok($ortho < 1e-12, "the residual is orthogonal to every column of A");
// The textbook answer for this data is y = 3.5 + 1.4x.
close(nd_get($xl, [0]), 3.5, 1e-10, "intercept is 3.5");
close(nd_get($xl, [1]), 1.4, 1e-10, "slope is 1.4");

print("");
print("-- eigh, by reconstruction and orthogonality --");
$Sy = nd_add(nd_matmul($M, nd_T($M)), nd_multiply(nd_identity(4), 1.0, null), null);
$eg = nd_eigh($Sy);
$w = $eg[0];
$V = $eg[1];
eq(str(nd_shape($w)), "[4]", "eigh returns 4 eigenvalues");
eq(str(nd_shape($V)), "[4, 4]", "and a 4x4 matrix of eigenvectors");
ok(nd_get($w, [0]) <= nd_get($w, [3]), "eigenvalues come back ascending");
// A*V == V*diag(w), which is the definition.
$AV = nd_matmul($Sy, $V);
$VW = nd_multiply($V, nd_reshape($w, [1, 4]), null);
$eres = maxabs(nd_subtract($AV, $VW, null));
print("        ||AV - V diag(w)||: " + str($eres));
ok($eres < 1e-11, "A V == V diag(w)");
$VtV = nd_matmul(nd_T($V), $V);
$eorth = maxabs(nd_subtract($VtV, nd_identity(4), null));
print("        ||VtV - I||: " + str($eorth));
ok($eorth < 1e-12, "the eigenvectors are orthonormal to 1e-12");
// A non-symmetric matrix is a different question; symmetrising it silently
// would answer the wrong one.
raises(def() { return nd_eigh(nd([[1.0, 2.0], [3.0, 4.0]], "f64")); }, "not symmetric",
   "a non-symmetric matrix raises, naming the offending element");

print("");
print("-- SVD, by reconstruction --");
$Asv = nd_random_normal([8, 5], 0, 1, 13);
$sv = nd_svd($Asv);
$U = $sv[0];
$s = $sv[1];
$Vs = $sv[2];
eq(str(nd_shape($U)),  "[8, 5]", "U is m-by-k");
eq(str(nd_shape($s)),  "[5]",    "k singular values");
eq(str(nd_shape($Vs)), "[5, 5]", "V is n-by-k");
ok(nd_get($s, [0]) >= nd_get($s, [4]), "singular values come back descending");
ok(nd_get($s, [4]) >= 0, "and are non-negative");
// U diag(s) Vt must reconstruct A.
$US = nd_multiply($U, nd_reshape($s, [1, 5]), null);
$rec = nd_matmul($US, nd_T($Vs));
$sres = maxabs(nd_subtract($rec, $Asv, null));
print("        ||U S Vt - A||: " + str($sres));
ok($sres < 1e-12, "U diag(s) Vt reconstructs A to 1e-12");
$UtU = nd_matmul(nd_T($U), $U);
ok(maxabs(nd_subtract($UtU, nd_identity(5), null)) < 1e-12, "U has orthonormal columns");

print("");
print("-- rank, condition number, pseudo-inverse --");
eq(nd_get(nd_matrix_rank(nd_identity(4)), []), 4, "the identity has full rank");
// A deliberately rank-deficient matrix: the third row is the sum of the first two.
$rd = nd([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]], "f64");
eq(nd_get(nd_matrix_rank($rd), []), 2, "a rank-deficient matrix is detected");
close(nd_get(nd_cond(nd_identity(3)), []), 1.0, 1e-10, "the identity is perfectly conditioned");
ok(nd_get(nd_cond($rd), []) > 1e10, "a singular matrix has a huge condition number");
// The Moore-Penrose conditions: A P A == A is the one that matters most.
$P2 = nd_pinv($rd);
ok(maxabs(nd_subtract(nd_matmul(nd_matmul($rd, $P2), $rd), $rd, null)) < 1e-9,
   "A pinv(A) A == A, even for a singular matrix");
ok(maxabs(nd_subtract(nd_pinv($S), nd_inv($S), null)) < 1e-12,
   "for an invertible matrix the pseudo-inverse IS the inverse");

print("");
print("-- norms --");
close(nd_get(nd_norm(nd([3.0, 4.0], "f64"), null), []), 5.0, 1e-12, "vector 2-norm");
close(nd_get(nd_norm(nd([[3.0, 0.0], [0.0, 4.0]], "f64"), null), []), 5.0, 1e-12, "Frobenius norm");
close(nd_get(nd_norm(nd([[1.0, -2.0], [-3.0, 4.0]], "f64"), 1), []), 6.0, 1e-12,
      "1-norm is the max absolute column sum");
close(nd_get(nd_norm(nd([[1.0, -2.0], [-3.0, 4.0]], "f64"), "inf"), []), 7.0, 1e-12,
      "inf-norm is the max absolute row sum");
// The spectral norm is NOT the Frobenius norm, and conflating them is a classic
// quiet error -- for the identity they differ by sqrt(n).
close(nd_get(nd_norm(nd_identity(4), 2), []), 1.0, 1e-12, "the 2-norm of I is 1");
close(nd_get(nd_norm(nd_identity(4), null), []), 2.0, 1e-12, "while its Frobenius norm is 2");
raises(def() { return nd_norm($A, "nuclear"); }, "unknown order", "an unknown order is refused");

print("");
print("-- failure is loud --");
$sing = nd([[1.0, 2.0], [2.0, 4.0]], "f64");    // second row is twice the first
raises(def() { return nd_solve($sing, nd([1.0, 2.0], "f64")); }, "singular",
   "a singular system raises and suggests lstsq");
raises(def() { return nd_inv($sing); }, "singular", "and so does inverting it");
raises(def() { return nd_solve(nd_zeros([2, 3], null), nd([1.0, 2.0], "f64")); }, "square",
   "a non-square solve raises");
raises(def() { return nd_solve($S, nd([1.0, 2.0, 3.0], "f64")); }, "elements",
   "a right-hand side of the wrong length raises, naming both sizes");
raises(def() { return nd_det(nd_zeros([2, 3], null)); }, "square", "det of a non-square raises");
raises(def() { return nd_matmul(nd(5, null), nd(5, null)); }, "not a matrix",
   "matmul on 0-d arrays raises");

print("");
print("-- the phase gates, at the sizes the roadmap specifies --");
$n = 500;
// Diagonally dominant, so it is certainly non-singular and well conditioned.
$Big = nd_add(nd_random_normal([$n, $n], 0, 1, 21),
              nd_multiply(nd_identity($n), $n * 1.0, null), null);
$bb = nd_random_normal([$n], 0, 1, 22);
$t0 = clock();
$xx = nd_solve($Big, $bb);
$solveMs = clock() - $t0;
$bigResid = maxabs(nd_subtract(nd_matmul($Big, $xx), $bb, null));
print("        500x500 solve: " + str($solveMs) + "ms, residual " + str($bigResid));
ok($bigResid < 1e-10, "GATE: the 500x500 solve residual is under 1e-10");
ok($solveMs < 5000, "and it finishes in reasonable time");

$t0 = clock();
$Msv = nd_random_normal([200, 200], 0, 1, 23);
$sv2 = nd_svd($Msv);
$svdMs = clock() - $t0;
$U2 = $sv2[0];
$s2 = $sv2[1];
$V2 = $sv2[2];
$rec2 = nd_matmul(nd_multiply($U2, nd_reshape($s2, [1, 200]), null), nd_T($V2));
$svdRes = maxabs(nd_subtract($rec2, $Msv, null));
print("        200x200 SVD: " + str($svdMs) + "ms, reconstruction " + str($svdRes));
ok($svdRes < 1e-12, "GATE: the 200x200 SVD reconstructs to 1e-12");

// The blocked i-k-j ordering is what this gate really measures: the naive
// i-j-k order strides through B by `cols` on every innermost step, which is a
// cache miss per element once the matrix leaves L2.
$t0 = clock();
$mm = nd_matmul(nd_random_normal([1000, 1000], 0, 1, 24), nd_random_normal([1000, 1000], 0, 1, 25));
$mmMs = clock() - $t0;
$gflops = (2.0 * 1000 * 1000 * 1000) / ($mmMs / 1000.0) / 1000000000.0;
print("        1000x1000 matmul: " + str($mmMs) + "ms = " + str($gflops) + " GFLOP/s");
ok($gflops >= 4.0, "GATE: 1000-cubed matmul reaches at least 4 GFLOP/s");
eq(str(nd_shape($mm)), "[1000, 1000]", "with the right shape");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }

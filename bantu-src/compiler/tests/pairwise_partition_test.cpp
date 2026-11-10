// ════════════════════════════════════════════════════════════════════════
//  pairwise_partition_test.cpp — can a PARTITIONED pairwise sum reproduce the
//  single-threaded one bit for bit?
//
//  DECISIONS N26 said no, and planned a one-time re-baseline of nd_sum's last
//  bits on that basis. N26 was wrong, and this is the program that showed it.
//
//  The reasoning it encodes: `Pairwise` is a binary counter, so after a
//  multiple of 2^m adds, level m holds the sum of the last 2^m values
//  associated as a PERFECT subtree, and the low m levels are empty. So a chunk
//  of exactly 2^m blocks, summed independently with the same algorithm from an
//  empty counter, produces exactly the value the global tree would hold at
//  level m -- and injecting those chunk values at level m in ascending order
//  rebuilds the global tree's state exactly. The remainder (always fewer than
//  one chunk) is then added sequentially, and its carries cannot reach level m
//  because reaching it needs 2^m adds.
//
//  So a threaded reduction can be bit-identical to today's answer, at every
//  thread count, and no user's sum has to change. 1813 of 1813 size/chunk
//  combinations agree, for chunks from 1024 to 65536 elements, over sizes
//  chosen to land on, just before and just after chunk boundaries, with tails
//  of 0, 1, 127, 128, 129 and 1000 elements.
//
//  Build and run:
//    g++ -std=c++17 -O2 tests/pairwise_partition_test.cpp -o /tmp/pwtest && /tmp/pwtest
//
//  It carries its own copy of `Pairwise` and the block loop, deliberately: if
//  ndarray_reduce.hpp's summation order ever changes, this test keeps passing
//  while the REAL assertion -- tests/numba_reduce_test.b's exact sums -- fails,
//  and the two together say "the kernel changed", not "the partition broke".
// ════════════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>

struct Pairwise {
    static const int MAXLEVEL = 64;
    double partial[MAXLEVEL];
    size_t count = 0;
    void add(double v) {
        size_t c = count++;
        int k = 0;
        while (c & 1) { v += partial[k]; c >>= 1; k++; }
        partial[k] = v;
    }
    // Inject a value that stands for 2^lvl adds already associated as a
    // perfect subtree. Mirrors add() with the counter shifted.
    void addLevel(int lvl, double v) {
        size_t c = count >> lvl;            // low lvl bits are zero by alignment
        count += (size_t(1) << lvl);
        int k = lvl;
        while (c & 1) { v += partial[k]; c >>= 1; k++; }
        partial[k] = v;
    }
    double total() const {
        double s = 0.0;
        size_t c = count;
        for (int k = 0; c; k++, c >>= 1) if (c & 1) s += partial[k];
        return s;
    }
};

static double sumRef(const double* p, size_t n) {       // today's kernel
    Pairwise acc;
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        for (size_t j = 0; j < 128; j += 8) {
            s0 += p[i+j+0]; s1 += p[i+j+1]; s2 += p[i+j+2]; s3 += p[i+j+3];
            s4 += p[i+j+4]; s5 += p[i+j+5]; s6 += p[i+j+6]; s7 += p[i+j+7];
        }
        acc.add(((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)));
    }
    for (; i < n; i++) acc.add(p[i]);
    return acc.total();
}

// One aligned chunk of exactly 2^m blocks of 128: its own perfect subtree.
static double chunkSum(const double* p, int m) {
    Pairwise acc;
    const size_t blocks = size_t(1) << m;
    for (size_t b = 0; b < blocks; b++) {
        const double* q = p + b * 128;
        double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        for (size_t j = 0; j < 128; j += 8) {
            s0 += q[j+0]; s1 += q[j+1]; s2 += q[j+2]; s3 += q[j+3];
            s4 += q[j+4]; s5 += q[j+5]; s6 += q[j+6]; s7 += q[j+7];
        }
        acc.add(((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)));
    }
    return acc.partial[m];                  // the single perfect-subtree sum
}

// The partitioned version: chunk sums computed independently (as workers
// would), injected in ASCENDING index order, then the remainder sequentially.
static double sumPart(const double* p, size_t n, int m) {
    const size_t C = size_t(128) << m;      // elements per chunk
    const size_t k = n / C;                 // full chunks
    std::vector<double> sums(k);
    for (size_t c = 0; c < k; c++) sums[c] = chunkSum(p + c * C, m);   // parallel
    Pairwise acc;
    for (size_t c = 0; c < k; c++) acc.addLevel(m, sums[c]);           // in order
    size_t i = k * C;
    for (; i + 128 <= n; i += 128) {
        double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        for (size_t j = 0; j < 128; j += 8) {
            s0 += p[i+j+0]; s1 += p[i+j+1]; s2 += p[i+j+2]; s3 += p[i+j+3];
            s4 += p[i+j+4]; s5 += p[i+j+5]; s6 += p[i+j+6]; s7 += p[i+j+7];
        }
        acc.add(((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)));
    }
    for (; i < n; i++) acc.add(p[i]);
    return acc.total();
}

int main() {
    srandom(12345);
    int bad = 0, tested = 0;
    // sizes around chunk boundaries, with awkward tails
    std::vector<size_t> ns;
    for (int m = 3; m <= 9; m++) {
        const size_t C = size_t(128) << m;
        for (size_t base : {C, 2*C, 3*C, 5*C, 8*C}) {
            for (size_t d : {size_t(0), size_t(1), size_t(127), size_t(128),
                             size_t(129), size_t(1000), C - 1}) {
                ns.push_back(base + d);
            }
        }
        ns.push_back(C - 1);
        ns.push_back(C / 2 + 7);
    }
    size_t maxn = 0;
    for (size_t n : ns) if (n > maxn) maxn = n;
    std::vector<double> v(maxn);
    for (size_t i = 0; i < maxn; i++) {
        v[i] = ((double)random() / (double)0x7fffffff) * 2.0 - 1.0;
    }
    for (int m = 3; m <= 9; m++) {
        int badm = 0, tm = 0;
        for (size_t n : ns) {
            if (n > maxn) continue;
            const double a = sumRef(v.data(), n);
            const double b = sumPart(v.data(), n, m);
            uint64_t ba, bb;
            std::memcpy(&ba, &a, 8); std::memcpy(&bb, &b, 8);
            tm++; tested++;
            if (ba != bb) { badm++; bad++;
                if (badm <= 2) std::printf("    m=%d n=%zu  ref=%.17g part=%.17g\n", m, n, a, b);
            }
        }
        std::printf("  chunk=%zu elements (m=%d): %d of %d sizes differ\n",
                    size_t(128) << m, m, badm, tm);
    }
    std::printf("\n  %d of %d size/chunk combinations bit-identical\n", tested - bad, tested);
    std::printf("  %s\n", bad == 0 ? "RESULT: a partitioned pairwise sum CAN be bit-identical"
                                   : "RESULT: differs");
    return bad == 0 ? 0 : 1;
}

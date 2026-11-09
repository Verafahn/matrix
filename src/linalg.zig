//! Linear algebra algorithms for floating-point matrices.
//!
//! All functions accept the matrix types produced by `MatrixF`. Dimensions are
//! compile-time constants, so incompatible shapes are rejected by the compiler.
//! Numerical routines that can fail (singular systems, rank-deficient QR,
//! non-positive-definite Cholesky input, or a non-converged eigenvalue
//! iteration) return `null`.
//!
//! ```zig
//! const matrix = @import("matrix");
//! const linalg = matrix.linalg;
//!
//! const A = matrix.MatrixF(f64, 2, 2);
//! const a = A.fromRows(.{ .{ 4, 7 }, .{ 2, 6 } });
//! const x = linalg.solve(a, @Vector(2, f64){ 1, 2 });
//! ```

const std = @import("std");
const MatrixF = @import("MatrixF.zig").MatrixF;

fn validateFloat(comptime T: type, comptime who: []const u8) void {
    if (@typeInfo(T.Ele) != .float) {
        @compileError(who ++ " requires a floating-point matrix");
    }
}

fn validateFloatSquare(comptime T: type, comptime who: []const u8) void {
    validateFloat(T, who);
    if (T.row != T.col) {
        @compileError(who ++ " requires a square matrix");
    }
}

/// Result of an LU decomposition with partial pivoting.
///
/// The decomposition satisfies `P * A = L * U`, where `permutation` describes
/// the row permutation `P`.
pub fn Lu(comptime F: type, comptime n: comptime_int) type {
    return struct {
        const Self = @This();

        /// Element type.
        pub const Ele = F;
        /// Matrix dimension.
        pub const size = n;

        /// Unit lower-triangular factor.
        lower: MatrixF(F, n, n),
        /// Upper-triangular factor.
        upper: MatrixF(F, n, n),
        /// `permutation[i]` is the original row that became row `i` of `P * A`.
        permutation: [n]usize,
        /// Sign of the permutation (`1` or `-1`).
        parity: i8,

        /// Solves `A * x = b` using the stored factors.
        pub fn solve(self: Self, b: @Vector(n, F)) @Vector(n, F) {
            const rhs: [n]F = b;

            var y: [n]F = undefined;
            for (0..n) |i| {
                var sum = rhs[self.permutation[i]];
                for (0..i) |j| {
                    sum -= self.lower.at(i, j) * y[j];
                }
                y[i] = sum;
            }

            var x: [n]F = undefined;
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum = y[i];
                for (i + 1..n) |j| {
                    sum -= self.upper.at(i, j) * x[j];
                }
                x[i] = sum / self.upper.at(i, i);
            }
            return x;
        }

        /// Solves `A * X = B` for a matrix right-hand side.
        pub fn solveMatrix(self: Self, b: anytype) @TypeOf(b) {
            const B = @TypeOf(b);
            comptime {
                if (B.Ele != F) @compileError("matrix element types must match");
                if (B.row != n) @compileError("right-hand side has incompatible dimensions");
            }

            var data: [n * B.col]F = undefined;
            for (0..B.col) |c| {
                const column = self.solve(b.getCol(c));
                const values: [n]F = column;
                for (0..n) |r| {
                    data[r * B.col + c] = values[r];
                }
            }
            return B.fromArray(data);
        }

        /// Returns the determinant represented by this decomposition.
        pub fn determinant(self: Self) F {
            var value: F = if (self.parity < 0) -1 else 1;
            for (0..n) |i| {
                value *= self.upper.at(i, i);
            }
            return value;
        }
    };
}

/// Computes an LU decomposition with partial pivoting.
///
/// Returns `null` if the matrix is singular.
pub fn lu(a: anytype) ?Lu(@TypeOf(a).Ele, @TypeOf(a).row) {
    const A = @TypeOf(a);
    comptime validateFloatSquare(A, "lu");

    const F = A.Ele;
    const n = A.row;

    var factors: [n * n]F = a.toArray();
    var permutation: [n]usize = undefined;
    for (0..n) |i| permutation[i] = i;
    var parity: i8 = 1;

    for (0..n) |k| {
        var pivot_row = k;
        var pivot_abs = @abs(factors[k * n + k]);
        for (k + 1..n) |r| {
            const candidate = @abs(factors[r * n + k]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = r;
            }
        }
        if (pivot_abs == 0) return null;

        if (pivot_row != k) {
            for (0..n) |c| {
                const tmp = factors[k * n + c];
                factors[k * n + c] = factors[pivot_row * n + c];
                factors[pivot_row * n + c] = tmp;
            }
            const tmp = permutation[k];
            permutation[k] = permutation[pivot_row];
            permutation[pivot_row] = tmp;
            parity = -parity;
        }

        const pivot = factors[k * n + k];
        for (k + 1..n) |r| {
            factors[r * n + k] /= pivot;
            const multiplier = factors[r * n + k];
            for (k + 1..n) |c| {
                factors[r * n + c] -= multiplier * factors[k * n + c];
            }
        }
    }

    var lower_data: [n * n]F = @splat(0);
    var upper_data: [n * n]F = @splat(0);
    for (0..n) |r| {
        for (0..n) |c| {
            if (r > c) {
                lower_data[r * n + c] = factors[r * n + c];
            } else {
                if (r == c) lower_data[r * n + c] = 1;
                upper_data[r * n + c] = factors[r * n + c];
            }
        }
    }

    return .{
        .lower = MatrixF(F, n, n).fromArray(lower_data),
        .upper = MatrixF(F, n, n).fromArray(upper_data),
        .permutation = permutation,
        .parity = parity,
    };
}

/// Solves the square linear system `A * x = b`.
///
/// Returns `null` if `A` is singular.
pub fn solve(
    a: anytype,
    b: @Vector(@TypeOf(a).row, @TypeOf(a).Ele),
) ?@Vector(@TypeOf(a).row, @TypeOf(a).Ele) {
    const decomposition = lu(a) orelse return null;
    return decomposition.solve(b);
}

/// Solves the square linear system `A * X = B`.
///
/// Returns `null` if `A` is singular.
pub fn solveMatrix(a: anytype, b: anytype) ?@TypeOf(b) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    comptime {
        validateFloatSquare(A, "solveMatrix");
        if (B.Ele != A.Ele) @compileError("matrix element types must match");
        if (B.row != A.row) @compileError("right-hand side has incompatible dimensions");
    }

    const decomposition = lu(a) orelse return null;
    return decomposition.solveMatrix(b);
}

/// Returns the submatrix obtained by removing row `row_index` and column
/// `col_index`.
pub fn minor(a: anytype, row_index: usize, col_index: usize) MatrixF(
    @TypeOf(a).Ele,
    @TypeOf(a).row - 1,
    @TypeOf(a).col - 1,
) {
    const A = @TypeOf(a);
    comptime {
        validateFloat(A, "minor");
        if (A.row < 2 or A.col < 2) {
            @compileError("minor requires at least 2 rows and 2 columns");
        }
    }
    std.debug.assert(row_index < A.row and col_index < A.col);

    const F = A.Ele;
    const m = A.row - 1;
    const n = A.col - 1;
    var data: [m * n]F = undefined;

    var dest_row: usize = 0;
    for (0..A.row) |r| {
        if (r == row_index) continue;
        var dest_col: usize = 0;
        for (0..A.col) |c| {
            if (c == col_index) continue;
            data[dest_row * n + dest_col] = a.at(r, c);
            dest_col += 1;
        }
        dest_row += 1;
    }
    return MatrixF(F, m, n).fromArray(data);
}

/// Returns the cofactor of a square matrix at row `r`, column `c`.
pub fn cofactor(a: anytype, r: usize, c: usize) @TypeOf(a).Ele {
    const A = @TypeOf(a);
    comptime {
        validateFloatSquare(A, "cofactor");
        if (A.row < 2) @compileError("cofactor requires a matrix of size at least 2");
    }

    const value = minor(a, r, c).determinant();
    return if ((r + c) % 2 == 0) value else -value;
}

/// Returns the adjugate (classical adjoint) of a square matrix.
///
/// For every matrix, `A * adjugate(A) == determinant(A) * I`.
pub fn adjugate(a: anytype) MatrixF(@TypeOf(a).Ele, @TypeOf(a).row, @TypeOf(a).col) {
    const A = @TypeOf(a);
    comptime validateFloatSquare(A, "adjugate");

    const F = A.Ele;
    const n = A.row;
    if (n == 1) return MatrixF(F, 1, 1).fromArray(.{1});

    var data: [n * n]F = undefined;
    for (0..n) |r| {
        for (0..n) |c| {
            data[c * n + r] = cofactor(a, r, c);
        }
    }
    return MatrixF(F, n, n).fromArray(data);
}

/// Returns `true` when `A` is symmetric within `tolerance`.
pub fn isSymmetric(a: anytype, tolerance: @TypeOf(a).Ele) bool {
    const A = @TypeOf(a);
    comptime validateFloatSquare(A, "isSymmetric");
    std.debug.assert(tolerance >= 0);

    for (0..A.row) |r| {
        for (0..A.col) |c| {
            if (@abs(a.at(r, c) - a.at(c, r)) > tolerance) return false;
        }
    }
    return true;
}

/// Computes the Cholesky factor `L` of a symmetric positive-definite matrix,
/// such that `A == L * L^T`.
///
/// Only the lower triangle of `A` is read; the input is expected to be
/// symmetric. Returns `null` when `A` is not positive definite.
pub fn cholesky(a: anytype) ?MatrixF(@TypeOf(a).Ele, @TypeOf(a).row, @TypeOf(a).col) {
    const A = @TypeOf(a);
    comptime validateFloatSquare(A, "cholesky");

    const F = A.Ele;
    const n = A.row;
    var lower_data: [n * n]F = @splat(0);

    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum = a.at(i, j);
            for (0..j) |k| {
                sum -= lower_data[i * n + k] * lower_data[j * n + k];
            }

            if (i == j) {
                if (!(sum > 0)) return null;
                lower_data[i * n + i] = @sqrt(sum);
            } else {
                lower_data[i * n + j] = sum / lower_data[j * n + j];
            }
        }
    }
    return MatrixF(F, n, n).fromArray(lower_data);
}

/// Result of a thin QR decomposition.
///
/// `Q` is `m x n` with orthonormal columns and `R` is `n x n` upper
/// triangular, satisfying `A == Q * R`.
pub fn Qr(comptime F: type, comptime m: comptime_int, comptime n: comptime_int) type {
    return struct {
        const Self = @This();

        /// Element type.
        pub const Ele = F;
        /// Number of rows.
        pub const rows = m;
        /// Number of columns.
        pub const cols = n;

        /// Orthonormal columns.
        q: MatrixF(F, m, n),
        /// Upper-triangular factor.
        r: MatrixF(F, n, n),

        /// Solves `A * x = b` for square systems or the least-squares problem
        /// `min ||A * x - b||` for overdetermined systems.
        pub fn solve(self: Self, b: @Vector(m, F)) @Vector(n, F) {
            const projected = self.q.transpose().mulVec(b);
            const rhs: [n]F = projected;

            var x: [n]F = undefined;
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum = rhs[i];
                for (i + 1..n) |j| {
                    sum -= self.r.at(i, j) * x[j];
                }
                x[i] = sum / self.r.at(i, i);
            }
            return x;
        }
    };
}

/// Computes a thin QR decomposition using modified Gram-Schmidt.
///
/// Requires `rows >= cols`. Returns `null` when the columns are linearly
/// dependent.
pub fn qr(a: anytype) ?Qr(@TypeOf(a).Ele, @TypeOf(a).row, @TypeOf(a).col) {
    const A = @TypeOf(a);
    comptime {
        validateFloat(A, "qr");
        if (A.row < A.col) @compileError("qr requires rows >= columns");
    }

    const F = A.Ele;
    const m = A.row;
    const n = A.col;
    var q_data: [m * n]F = undefined;
    var r_data: [n * n]F = @splat(0);

    for (0..n) |j| {
        var v: [m]F = undefined;
        for (0..m) |i| v[i] = a.at(i, j);

        for (0..j) |i| {
            var dot: F = 0;
            for (0..m) |k| {
                dot += q_data[k * n + i] * v[k];
            }
            r_data[i * n + j] = dot;
            for (0..m) |k| {
                v[k] -= dot * q_data[k * n + i];
            }
        }

        var norm_squared: F = 0;
        for (0..m) |k| norm_squared += v[k] * v[k];
        const norm = @sqrt(norm_squared);
        if (norm == 0) return null;

        r_data[j * n + j] = norm;
        for (0..m) |k| q_data[k * n + j] = v[k] / norm;
    }

    return .{
        .q = MatrixF(F, m, n).fromArray(q_data),
        .r = MatrixF(F, n, n).fromArray(r_data),
    };
}

/// Solves the least-squares problem `min ||A * x - b||`.
///
/// Requires `A.row >= A.col`. Returns `null` if `A` does not have full column
/// rank.
pub fn leastSquares(
    a: anytype,
    b: @Vector(@TypeOf(a).row, @TypeOf(a).Ele),
) ?@Vector(@TypeOf(a).col, @TypeOf(a).Ele) {
    const decomposition = qr(a) orelse return null;
    return decomposition.solve(b);
}

/// Result of a symmetric eigenvalue decomposition.
pub fn Eigen(comptime F: type, comptime n: comptime_int) type {
    return struct {
        const Self = @This();

        /// Element type.
        pub const Ele = F;
        /// Matrix dimension.
        pub const size = n;

        /// Eigenvalues sorted in ascending order.
        values: @Vector(n, F),
        /// Eigenvectors stored by column, in the same order as `values`.
        vectors: MatrixF(F, n, n),

        /// Returns the eigenvector at column `i`.
        pub fn eigenvector(self: Self, i: usize) @Vector(n, F) {
            std.debug.assert(i < n);
            return self.vectors.getCol(i);
        }
    };
}

/// Computes the eigenvalues and eigenvectors of a symmetric matrix using the
/// Jacobi rotation algorithm.
///
/// `tolerance` is an absolute convergence tolerance. Eigenvalues are sorted in
/// ascending order. Returns `null` if `A` is not symmetric within `tolerance`
/// or if the iteration did not converge.
pub fn eigenSymmetric(a: anytype, tolerance: @TypeOf(a).Ele) ?Eigen(@TypeOf(a).Ele, @TypeOf(a).row) {
    const A = @TypeOf(a);
    comptime validateFloatSquare(A, "eigenSymmetric");
    std.debug.assert(tolerance >= 0);

    if (!isSymmetric(a, tolerance)) return null;

    const F = A.Ele;
    const n = A.row;
    var matrix: [n][n]F = undefined;
    var vectors: [n][n]F = undefined;

    for (0..n) |r| {
        for (0..n) |c| matrix[r][c] = a.at(r, c);
        for (0..n) |c| vectors[r][c] = if (r == c) 1 else 0;
    }

    const max_iterations: usize = 100 * n * n;
    var iteration: usize = 0;
    var converged = n == 1;

    while (n > 1 and iteration < max_iterations) : (iteration += 1) {
        var p: usize = 0;
        var q: usize = 1;
        var largest: F = 0;

        for (0..n) |i| {
            for (i + 1..n) |j| {
                const value = @abs(matrix[i][j]);
                if (value > largest) {
                    largest = value;
                    p = i;
                    q = j;
                }
            }
        }

        if (largest <= tolerance) {
            converged = true;
            break;
        }

        const app = matrix[p][p];
        const aqq = matrix[q][q];
        const apq = matrix[p][q];
        const theta = (aqq - app) / (2 * apq);
        const t = if (theta >= 0)
            1 / (theta + @sqrt(theta * theta + 1))
        else
            1 / (theta - @sqrt(theta * theta + 1));
        const c = 1 / @sqrt(t * t + 1);
        const s = t * c;

        for (0..n) |i| {
            if (i == p or i == q) continue;
            const aip = matrix[i][p];
            const aiq = matrix[i][q];
            matrix[i][p] = c * aip - s * aiq;
            matrix[p][i] = matrix[i][p];
            matrix[i][q] = s * aip + c * aiq;
            matrix[q][i] = matrix[i][q];
        }

        matrix[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0;
        matrix[q][p] = 0;

        for (0..n) |i| {
            const vip = vectors[i][p];
            const viq = vectors[i][q];
            vectors[i][p] = c * vip - s * viq;
            vectors[i][q] = s * vip + c * viq;
        }
    }

    if (!converged) return null;

    var order: [n]usize = undefined;
    for (0..n) |i| order[i] = i;
    for (0..n) |i| {
        var smallest = i;
        for (i + 1..n) |j| {
            if (matrix[order[j]][order[j]] < matrix[order[smallest]][order[smallest]]) {
                smallest = j;
            }
        }
        if (smallest != i) {
            const tmp = order[i];
            order[i] = order[smallest];
            order[smallest] = tmp;
        }
    }

    var values: [n]F = undefined;
    var sorted_vectors: [n * n]F = undefined;
    for (0..n) |j| {
        values[j] = matrix[order[j]][order[j]];
        for (0..n) |i| {
            sorted_vectors[i * n + j] = vectors[i][order[j]];
        }
    }

    return .{
        .values = values,
        .vectors = MatrixF(F, n, n).fromArray(sorted_vectors),
    };
}

/// Result of a reduced row echelon form computation.
pub fn Rref(comptime F: type, comptime m: comptime_int, comptime n: comptime_int) type {
    return struct {
        const Self = @This();

        /// Element type.
        pub const Ele = F;
        /// Number of rows.
        pub const rows = m;
        /// Number of columns.
        pub const cols = n;
        /// Maximum possible number of pivots.
        pub const pivot_capacity = @min(m, n);

        /// The reduced row echelon form.
        matrix: MatrixF(F, m, n),
        /// Rank of the matrix.
        rank: usize,
        /// Pivot column indices; only the first `rank` entries are valid.
        pivot_columns: [pivot_capacity]usize,

        /// Returns the valid pivot columns.
        pub fn pivotColumns(self: *const Self) []const usize {
            return self.pivot_columns[0..self.rank];
        }
    };
}

/// Computes the reduced row echelon form of a matrix.
///
/// `tolerance` is an absolute threshold below which a pivot is treated as
/// zero.
pub fn rref(a: anytype, tolerance: @TypeOf(a).Ele) Rref(
    @TypeOf(a).Ele,
    @TypeOf(a).row,
    @TypeOf(a).col,
) {
    const A = @TypeOf(a);
    comptime validateFloat(A, "rref");
    std.debug.assert(tolerance >= 0);

    const F = A.Ele;
    const m = A.row;
    const n = A.col;
    var data: [m * n]F = a.toArray();
    var pivot_columns: [@min(m, n)]usize = undefined;
    for (0..@min(m, n)) |i| pivot_columns[i] = 0;

    var current_row: usize = 0;
    var pivot_count: usize = 0;

    for (0..n) |col| {
        if (current_row == m) break;

        var pivot_row = current_row;
        var largest = @abs(data[current_row * n + col]);
        for (current_row + 1..m) |r| {
            const candidate = @abs(data[r * n + col]);
            if (candidate > largest) {
                largest = candidate;
                pivot_row = r;
            }
        }
        if (largest <= tolerance) continue;

        if (pivot_row != current_row) {
            for (0..n) |c| {
                const tmp = data[current_row * n + c];
                data[current_row * n + c] = data[pivot_row * n + c];
                data[pivot_row * n + c] = tmp;
            }
        }

        const pivot = data[current_row * n + col];
        for (0..n) |c| data[current_row * n + c] /= pivot;

        for (0..m) |r| {
            if (r == current_row) continue;
            const factor = data[r * n + col];
            if (factor == 0) continue;
            for (0..n) |c| {
                data[r * n + c] -= factor * data[current_row * n + c];
            }
            data[r * n + col] = 0;
        }

        pivot_columns[pivot_count] = col;
        pivot_count += 1;
        current_row += 1;
    }

    return .{
        .matrix = MatrixF(F, m, n).fromArray(data),
        .rank = pivot_count,
        .pivot_columns = pivot_columns,
    };
}

/// Returns the rank of a matrix.
///
/// `tolerance` is an absolute threshold below which a pivot is treated as
/// zero.
pub fn rank(a: anytype, tolerance: @TypeOf(a).Ele) usize {
    return rref(a, tolerance).rank;
}

/// A basis for the null space of a matrix.
///
/// The basis is stored in the first `nullity` entries of `vectors`.
pub fn NullSpace(comptime F: type, comptime n: comptime_int) type {
    return struct {
        const Self = @This();

        /// Element type.
        pub const Ele = F;
        /// Dimension of the vectors.
        pub const dim = n;
        /// Maximum possible nullity.
        pub const capacity = n;

        vectors: [n]@Vector(n, F),
        nullity: usize,

        /// Returns the non-zero basis vectors.
        pub fn basis(self: *const Self) []const @Vector(n, F) {
            return self.vectors[0..self.nullity];
        }
    };
}

/// Computes a basis for the null space of a matrix.
///
/// `tolerance` is an absolute threshold below which a pivot is treated as
/// zero.
pub fn nullSpace(a: anytype, tolerance: @TypeOf(a).Ele) NullSpace(@TypeOf(a).Ele, @TypeOf(a).col) {
    const A = @TypeOf(a);
    comptime validateFloat(A, "nullSpace");
    const F = A.Ele;
    const n = A.col;

    const reduced = rref(a, tolerance);
    var vectors: [n]@Vector(n, F) = undefined;
    for (0..n) |i| vectors[i] = @splat(0);

    var count: usize = 0;
    for (0..n) |col| {
        var is_pivot = false;
        for (0..reduced.rank) |i| {
            if (reduced.pivot_columns[i] == col) {
                is_pivot = true;
                break;
            }
        }
        if (is_pivot) continue;

        var values: [n]F = @splat(0);
        values[col] = 1;
        for (0..reduced.rank) |i| {
            values[reduced.pivot_columns[i]] = -reduced.matrix.at(i, col);
        }
        vectors[count] = values;
        count += 1;
    }

    return .{ .vectors = vectors, .nullity = count };
}

fn expectApproxVector(
    comptime n: comptime_int,
    expected: @Vector(n, f64),
    actual: @Vector(n, f64),
    tolerance: f64,
) !void {
    const e: [n]f64 = expected;
    const a: [n]f64 = actual;
    for (0..n) |i| {
        try std.testing.expect(std.math.approxEqAbs(f64, e[i], a[i], tolerance));
    }
}

test "LU decomposition and linear solves" {
    const Mat = MatrixF(f64, 3, 3);
    const a = Mat.fromRows(.{
        .{ 2, -1, 1 },
        .{ 1, 3, 2 },
        .{ 1, -1, 2 },
    });

    const decomposition = lu(a) orelse return error.TestUnexpectedResult;

    var permuted: [3][3]f64 = undefined;
    for (0..3) |r| {
        for (0..3) |c| permuted[r][c] = a.at(decomposition.permutation[r], c);
    }
    try std.testing.expect(decomposition.lower.mul(decomposition.upper).approxEql(Mat.fromRows(permuted), 1e-12));
    try std.testing.expect(std.math.approxEqAbs(f64, a.determinant(), decomposition.determinant(), 1e-12));

    const expected: @Vector(3, f64) = .{ 1, 2, 3 };
    const b = a.mulVec(expected);
    const x = solve(a, b) orelse return error.TestUnexpectedResult;
    try expectApproxVector(3, expected, x, 1e-10);

    const solution_matrix = solveMatrix(a, Mat.identity()) orelse return error.TestUnexpectedResult;
    try std.testing.expect(solution_matrix.approxEql(a.inverse().?, 1e-10));

    const singular = MatrixF(f64, 2, 2).fromRows(.{
        .{ 1, 2 },
        .{ 2, 4 },
    });
    try std.testing.expect(lu(singular) == null);
    try std.testing.expect(solve(singular, @Vector(2, f64){ 1, 2 }) == null);

    const scalar = MatrixF(f64, 1, 1).fromArray(.{2});
    const scalar_solution = solve(scalar, @Vector(1, f64){4}) orelse return error.TestUnexpectedResult;
    try expectApproxVector(1, .{2}, scalar_solution, 1e-12);
}

test "Cholesky decomposition" {
    const Mat = MatrixF(f64, 3, 3);
    const a = Mat.fromRows(.{
        .{ 4, 12, -16 },
        .{ 12, 37, -43 },
        .{ -16, -43, 98 },
    });
    const expected = Mat.fromRows(.{
        .{ 2, 0, 0 },
        .{ 6, 1, 0 },
        .{ -8, 5, 3 },
    });

    const lower = cholesky(a) orelse return error.TestUnexpectedResult;
    try std.testing.expect(lower.approxEql(expected, 1e-12));
    try std.testing.expect(lower.mul(lower.transpose()).approxEql(a, 1e-12));

    const not_positive_definite = MatrixF(f64, 2, 2).fromRows(.{
        .{ 1, 2 },
        .{ 2, 1 },
    });
    try std.testing.expect(cholesky(not_positive_definite) == null);
}

test "QR decomposition and least squares" {
    const Mat = MatrixF(f64, 4, 2);
    const a = Mat.fromRows(.{
        .{ 1, 0 },
        .{ 1, 1 },
        .{ 1, 2 },
        .{ 1, 3 },
    });

    const decomposition = qr(a) orelse return error.TestUnexpectedResult;
    const identity = MatrixF(f64, 2, 2).identity();
    try std.testing.expect(decomposition.q.transpose().mul(decomposition.q).approxEql(identity, 1e-12));
    try std.testing.expect(decomposition.q.mul(decomposition.r).approxEql(a, 1e-12));

    const b: @Vector(4, f64) = .{ 1, 3, 5, 7 };
    const expected: @Vector(2, f64) = .{ 1, 2 };
    try expectApproxVector(2, expected, decomposition.solve(b), 1e-10);
    try expectApproxVector(2, expected, leastSquares(a, b) orelse return error.TestUnexpectedResult, 1e-10);

    const deficient = MatrixF(f64, 2, 2).fromRows(.{
        .{ 1, 2 },
        .{ 2, 4 },
    });
    try std.testing.expect(qr(deficient) == null);
    try std.testing.expect(leastSquares(deficient, @Vector(2, f64){ 1, 2 }) == null);
}

test "symmetric eigenvalues and eigenvectors" {
    const Mat = MatrixF(f64, 2, 2);
    const a = Mat.fromRows(.{
        .{ 2, 1 },
        .{ 1, 2 },
    });

    const eigen = eigenSymmetric(a, 1e-12) orelse return error.TestUnexpectedResult;
    try expectApproxVector(2, .{ 1, 3 }, eigen.values, 1e-10);

    const values: [2]f64 = eigen.values;
    for (0..2) |i| {
        const vector = eigen.eigenvector(i);
        const product = a.mulVec(vector);
        const product_values: [2]f64 = product;
        const vector_values: [2]f64 = vector;
        for (0..2) |k| {
            try std.testing.expect(std.math.approxEqAbs(f64, values[i] * vector_values[k], product_values[k], 1e-10));
        }
    }
    try std.testing.expect(eigen.vectors.transpose().mul(eigen.vectors).approxEql(Mat.identity(), 1e-10));

    const Mat3 = MatrixF(f64, 3, 3);
    const a3 = Mat3.fromRows(.{
        .{ 2, 1, 0 },
        .{ 1, 2, 0 },
        .{ 0, 0, 5 },
    });
    const eigen3 = eigenSymmetric(a3, 1e-12) orelse return error.TestUnexpectedResult;
    try expectApproxVector(3, .{ 1, 3, 5 }, eigen3.values, 1e-9);
    try std.testing.expect(eigen3.vectors.transpose().mul(eigen3.vectors).approxEql(Mat3.identity(), 1e-9));

    const non_symmetric = Mat.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    try std.testing.expect(eigenSymmetric(non_symmetric, 1e-9) == null);
}

test "RREF, rank, and null space" {
    const Mat = MatrixF(f64, 3, 3);
    const a = Mat.fromRows(.{
        .{ 1, 2, 3 },
        .{ 2, 4, 6 },
        .{ 1, 1, 1 },
    });

    const reduced = rref(a, 1e-10);
    try std.testing.expectEqual(@as(usize, 2), reduced.rank);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, reduced.pivotColumns());
    try std.testing.expectEqual(@as(usize, 2), rank(a, 1e-10));

    const space = nullSpace(a, 1e-10);
    try std.testing.expectEqual(@as(usize, 1), space.nullity);
    try std.testing.expectEqual(@as(usize, 1), space.basis().len);
    try expectApproxVector(3, .{ 0, 0, 0 }, a.mulVec(space.basis()[0]), 1e-9);

    const identity = Mat.identity();
    try std.testing.expectEqual(@as(usize, 3), rank(identity, 1e-12));
    try std.testing.expectEqual(@as(usize, 0), nullSpace(identity, 1e-12).nullity);

    const zero = Mat.zero();
    try std.testing.expectEqual(@as(usize, 0), rank(zero, 1e-12));
    try std.testing.expectEqual(@as(usize, 3), nullSpace(zero, 1e-12).nullity);
}

test "minor, cofactor, and adjugate" {
    const Mat = MatrixF(f64, 2, 2);
    const a = Mat.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    try std.testing.expectEqual(@as(f64, 4), minor(a, 0, 0).at(0, 0));
    try std.testing.expectEqual(@as(f64, 4), cofactor(a, 0, 0));
    try std.testing.expectEqual(@as(f64, -3), cofactor(a, 0, 1));
    try std.testing.expectEqual(@as(f64, -2), cofactor(a, 1, 0));
    try std.testing.expectEqual(@as(f64, 1), cofactor(a, 1, 1));

    const expected = Mat.fromRows(.{
        .{ 4, -2 },
        .{ -3, 1 },
    });
    const adj = adjugate(a);
    try std.testing.expect(adj.approxEql(expected, 1e-12));
    try std.testing.expect(a.mul(adj).approxEql(Mat.identity().scale(a.determinant()), 1e-12));

    const scalar = MatrixF(f64, 1, 1).fromArray(.{5});
    try std.testing.expectEqual([1][1]f64{.{1}}, adjugate(scalar).toRows());
}

test "symmetry checks" {
    const Mat = MatrixF(f64, 2, 2);
    try std.testing.expect(isSymmetric(Mat.fromRows(.{ .{ 1, 2 }, .{ 2, 3 } }), 1e-12));
    try std.testing.expect(!isSymmetric(Mat.fromRows(.{ .{ 1, 2 }, .{ 3, 4 } }), 1e-12));
    try std.testing.expect(isSymmetric(Mat.fromRows(.{ .{ 1, 2 }, .{ 2.0001, 3 } }), 1e-3));
}

//! Shared implementation for the public `MatrixF` and `MatrixI` types.
//!
//! Matrices are stored in row-major order in a flat SIMD vector. The type is
//! intentionally value-based: every operation returns a new matrix.

const std = @import("std");

/// Creates a matrix with `row_cnt` rows and `col_cnt` columns.
///
/// This is an implementation detail used by `MatrixF` and `MatrixI`; prefer
/// those public constructors.
pub fn Matrix(comptime E: type, comptime row_cnt: comptime_int, comptime col_cnt: comptime_int) type {
    comptime {
        if (row_cnt <= 0 or col_cnt <= 0) {
            @compileError("matrix dimensions must be greater than zero");
        }
        switch (@typeInfo(E)) {
            .float, .int => {},
            else => @compileError("matrix element type must be a float or an integer, got " ++ @typeName(E)),
        }
    }

    return struct {
        const Self = @This();

        /// Element type.
        pub const Ele = E;
        /// Number of rows.
        pub const row = row_cnt;
        /// Number of columns.
        pub const col = col_cnt;
        /// Alias for `row`.
        pub const rows = row_cnt;
        /// Alias for `col`.
        pub const cols = col_cnt;
        /// Number of elements in the matrix.
        pub const len = row_cnt * col_cnt;
        /// Whether the matrix is square.
        pub const is_square = row_cnt == col_cnt;
        /// Number of elements on the main diagonal.
        pub const diagonal_len = @min(row_cnt, col_cnt);

        /// Flat, row-major storage.
        data: @Vector(len, Ele),

        /// Returns a matrix with every element set to `value`.
        pub fn init(value: Ele) Self {
            return .{ .data = @splat(value) };
        }

        /// Returns the zero matrix.
        pub fn zero() Self {
            return init(0);
        }

        /// Returns a matrix with every element set to one.
        pub fn one() Self {
            return init(1);
        }

        /// Returns the identity matrix.
        ///
        /// Only available for square matrices.
        pub fn identity() Self {
            comptime if (!is_square) @compileError("identity requires a square matrix");

            var flat: [len]Ele = @splat(0);
            inline for (0..row) |i| {
                flat[i * col + i] = 1;
            }
            return fromArray(flat);
        }

        /// Returns a matrix whose main diagonal contains `values` and whose
        /// other elements are zero.
        pub fn fromDiagonal(values: [diagonal_len]Ele) Self {
            var flat: [len]Ele = @splat(0);
            inline for (0..diagonal_len) |i| {
                flat[i * col + i] = values[i];
            }
            return fromArray(flat);
        }

        /// Creates a matrix from a flat, row-major array.
        pub fn fromArray(values: [len]Ele) Self {
            return .{ .data = values };
        }

        /// Creates a matrix from a flat, row-major slice.
        ///
        /// Returns `error.InvalidLength` when the slice length does not match
        /// the matrix dimensions.
        pub fn fromSlice(values: []const Ele) error{InvalidLength}!Self {
            if (values.len != len) return error.InvalidLength;
            var flat: [len]Ele = undefined;
            for (0..len) |i| flat[i] = values[i];
            return fromArray(flat);
        }

        /// Creates a matrix from nested row arrays.
        pub fn fromRows(values: [row][col]Ele) Self {
            var flat: [len]Ele = undefined;
            for (0..row) |r| {
                for (0..col) |c| {
                    flat[r * col + c] = values[r][c];
                }
            }
            return fromArray(flat);
        }

        /// Returns the elements as a flat, row-major array.
        pub fn toArray(self: Self) [len]Ele {
            return self.data;
        }

        /// Returns the elements as nested row arrays.
        pub fn toRows(self: Self) [row][col]Ele {
            const flat: [len]Ele = self.data;
            var values: [row][col]Ele = undefined;
            for (0..row) |r| {
                for (0..col) |c| {
                    values[r][c] = flat[r * col + c];
                }
            }
            return values;
        }

        /// Returns the element at row `r`, column `c`.
        pub fn at(self: Self, r: usize, c: usize) Ele {
            std.debug.assert(r < row and c < col);
            const flat: [len]Ele = self.data;
            return flat[r * col + c];
        }

        /// Sets the element at row `r`, column `c`.
        pub fn set(self: *Self, r: usize, c: usize, value: Ele) void {
            std.debug.assert(r < row and c < col);
            var flat: [len]Ele = self.data;
            flat[r * col + c] = value;
            self.data = flat;
        }

        /// Returns row `r` as a vector.
        pub fn getRow(self: Self, r: usize) @Vector(col, Ele) {
            std.debug.assert(r < row);
            const flat: [len]Ele = self.data;
            var values: [col]Ele = undefined;
            for (0..col) |c| values[c] = flat[r * col + c];
            return values;
        }

        /// Returns column `c` as a vector.
        pub fn getCol(self: Self, c: usize) @Vector(row, Ele) {
            std.debug.assert(c < col);
            const flat: [len]Ele = self.data;
            var values: [row]Ele = undefined;
            for (0..row) |r| values[r] = flat[r * col + c];
            return values;
        }

        /// Returns the transpose of this matrix.
        pub fn transpose(self: Self) Matrix(E, col, row) {
            const flat: [len]Ele = self.data;
            var values: [col * row]Ele = undefined;
            for (0..row) |r| {
                for (0..col) |c| {
                    values[c * row + r] = flat[r * col + c];
                }
            }
            return .{ .data = values };
        }

        /// Element-wise addition.
        pub fn add(self: Self, other: Self) Self {
            return .{ .data = self.data + other.data };
        }

        /// Element-wise subtraction.
        pub fn sub(self: Self, other: Self) Self {
            return .{ .data = self.data - other.data };
        }

        /// Element-wise negation.
        pub fn neg(self: Self) Self {
            return .{ .data = -self.data };
        }

        /// Multiplies every element by `factor`.
        pub fn scale(self: Self, factor: Ele) Self {
            return .{ .data = self.data * @as(@Vector(len, Ele), @splat(factor)) };
        }

        /// Divides every element by `divisor`.
        ///
        /// Only available for floating-point matrices; integer division is
        /// intentionally not provided because it silently truncates.
        pub fn divScalar(self: Self, divisor: Ele) Self {
            comptime {
                if (@typeInfo(Ele) != .float) {
                    @compileError("divScalar requires a floating-point matrix; use scale for integers");
                }
            }
            return .{ .data = self.data / @as(@Vector(len, Ele), @splat(divisor)) };
        }

        /// Element-wise (Hadamard) multiplication.
        pub fn hadamard(self: Self, other: Self) Self {
            return .{ .data = self.data * other.data };
        }

        /// Matrix multiplication.
        ///
        /// The number of columns in `self` must equal the number of rows in
        /// `other`. The result has `self.row` rows and `other.col` columns.
        pub fn mul(self: Self, other: anytype) Matrix(E, row, @TypeOf(other).col) {
            const Other = @TypeOf(other);
            comptime {
                if (Other.Ele != Ele) {
                    @compileError("matrix element types must match");
                }
                if (Other.row != col) {
                    @compileError("matrix dimensions are incompatible for multiplication");
                }
            }

            const lhs: [len]Ele = self.data;
            const rhs: [Other.len]Ele = other.data;
            var result: [row * Other.col]Ele = undefined;

            for (0..row) |r| {
                for (0..Other.col) |c| {
                    var sum: Ele = 0;
                    for (0..col) |k| {
                        sum += lhs[r * col + k] * rhs[k * Other.col + c];
                    }
                    result[r * Other.col + c] = sum;
                }
            }

            return .{ .data = result };
        }

        /// Multiplies this matrix by a column vector.
        pub fn mulVec(self: Self, vector: @Vector(col, Ele)) @Vector(row, Ele) {
            const flat: [len]Ele = self.data;
            const values: [col]Ele = vector;
            var result: [row]Ele = undefined;

            for (0..row) |r| {
                var sum: Ele = 0;
                for (0..col) |c| {
                    sum += flat[r * col + c] * values[c];
                }
                result[r] = sum;
            }

            return result;
        }

        /// Returns the sum of the main diagonal.
        ///
        /// Only available for square matrices.
        pub fn trace(self: Self) Ele {
            comptime if (!is_square) @compileError("trace requires a square matrix");

            const flat: [len]Ele = self.data;
            var sum: Ele = 0;
            for (0..row) |i| {
                sum += flat[i * col + i];
            }
            return sum;
        }

        /// Returns the determinant using LU decomposition with partial
        /// pivoting.
        ///
        /// Only available for square floating-point matrices.
        pub fn determinant(self: Self) Ele {
            comptime {
                if (!is_square) @compileError("determinant requires a square matrix");
                if (@typeInfo(Ele) != .float) @compileError("determinant requires a floating-point matrix");
            }

            const n = row;
            const flat: [len]Ele = self.data;
            var a: [n][n]Ele = undefined;
            for (0..n) |r| {
                for (0..n) |c| {
                    a[r][c] = flat[r * n + c];
                }
            }

            var det: Ele = 1;
            for (0..n) |k| {
                var pivot_row = k;
                var pivot_abs = @abs(a[k][k]);
                for (k + 1..n) |r| {
                    const candidate = @abs(a[r][k]);
                    if (candidate > pivot_abs) {
                        pivot_abs = candidate;
                        pivot_row = r;
                    }
                }

                if (pivot_abs == 0) return 0;

                if (pivot_row != k) {
                    const tmp = a[k];
                    a[k] = a[pivot_row];
                    a[pivot_row] = tmp;
                    det = -det;
                }

                const pivot = a[k][k];
                det *= pivot;
                for (k + 1..n) |r| {
                    const factor = a[r][k] / pivot;
                    for (k + 1..n) |c| {
                        a[r][c] -= factor * a[k][c];
                    }
                }
            }

            return det;
        }

        /// Returns the inverse using Gauss-Jordan elimination with partial
        /// pivoting, or `null` if the matrix is singular.
        ///
        /// Only available for square floating-point matrices.
        pub fn inverse(self: Self) ?Self {
            comptime {
                if (!is_square) @compileError("inverse requires a square matrix");
                if (@typeInfo(Ele) != .float) @compileError("inverse requires a floating-point matrix");
            }

            const n = row;
            const flat: [len]Ele = self.data;
            var augmented: [n][2 * n]Ele = undefined;

            for (0..n) |r| {
                for (0..n) |c| augmented[r][c] = flat[r * n + c];
                for (n..2 * n) |c| augmented[r][c] = if (c - n == r) 1 else 0;
            }

            for (0..n) |c| {
                var pivot_row = c;
                var pivot_abs = @abs(augmented[c][c]);
                for (c + 1..n) |r| {
                    const candidate = @abs(augmented[r][c]);
                    if (candidate > pivot_abs) {
                        pivot_abs = candidate;
                        pivot_row = r;
                    }
                }

                if (pivot_abs == 0) return null;

                if (pivot_row != c) {
                    const tmp = augmented[c];
                    augmented[c] = augmented[pivot_row];
                    augmented[pivot_row] = tmp;
                }

                const pivot = augmented[c][c];
                for (0..2 * n) |j| augmented[c][j] /= pivot;

                for (0..n) |r| {
                    if (r == c) continue;
                    const factor = augmented[r][c];
                    if (factor == 0) continue;
                    for (0..2 * n) |j| {
                        augmented[r][j] -= factor * augmented[c][j];
                    }
                }
            }

            var result: [len]Ele = undefined;
            for (0..n) |r| {
                for (0..n) |c| result[r * n + c] = augmented[r][n + c];
            }
            return fromArray(result);
        }

        /// Returns `true` when all elements are exactly equal.
        pub fn eql(self: Self, other: Self) bool {
            return @reduce(.And, self.data == other.data);
        }

        /// Returns `true` when every element differs by at most `tolerance`.
        ///
        /// Only available for floating-point matrices.
        pub fn approxEql(self: Self, other: Self, tolerance: Ele) bool {
            comptime {
                if (@typeInfo(Ele) != .float) {
                    @compileError("approxEql requires a floating-point matrix");
                }
            }
            std.debug.assert(tolerance >= 0);

            const difference = @abs(self.data - other.data);
            const limit: @Vector(len, Ele) = @splat(tolerance);
            return @reduce(.And, difference <= limit);
        }

        /// Formats the matrix as one bracketed row per line.
        pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            const flat: [len]Ele = self.data;
            for (0..row) |r| {
                if (r != 0) try writer.writeByte('\n');
                try writer.writeByte('[');
                for (0..col) |c| {
                    if (c != 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{flat[r * col + c]});
                }
                try writer.writeByte(']');
            }
        }
    };
}

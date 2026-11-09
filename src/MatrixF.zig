//! Floating-point matrices with statically known dimensions.

const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

/// Creates a row-major matrix with `row_cnt` rows and `col_cnt` columns whose
/// elements have the floating-point type `F`.
///
/// ```zig
/// const Mat = MatrixF(f32, 2, 2);
/// const m = Mat.fromRows(.{ .{ 1, 2 }, .{ 3, 4 } });
/// ```
pub fn MatrixF(comptime F: type, comptime row_cnt: comptime_int, comptime col_cnt: comptime_int) type {
    comptime {
        if (@typeInfo(F) != .float) {
            @compileError("MatrixF requires a floating-point element type, got " ++ @typeName(F));
        }
    }
    return Matrix(F, row_cnt, col_cnt);
}

test "construction, access, and conversion" {
    const Mat = MatrixF(f32, 2, 3);
    var m = Mat.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });

    try std.testing.expectEqual([2][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    }, m.toRows());
    try std.testing.expectEqual([6]f32{ 1, 2, 3, 4, 5, 6 }, m.toArray());
    try std.testing.expectEqual(@as(f32, 2), m.at(0, 1));
    try std.testing.expectEqual(@as(f32, 5), m.at(1, 1));

    m.set(1, 2, 9);
    try std.testing.expectEqual(@as(f32, 9), m.at(1, 2));

    try std.testing.expectEqual(@Vector(3, f32){ 1, 2, 3 }, m.getRow(0));
    try std.testing.expectEqual(@Vector(2, f32){ 3, 9 }, m.getCol(2));

    const flat = [_]f32{ 9, 8, 7, 6, 5, 4 };
    const from_slice = try Mat.fromSlice(&flat);
    try std.testing.expectEqual([6]f32{ 9, 8, 7, 6, 5, 4 }, from_slice.toArray());
    try std.testing.expectError(error.InvalidLength, Mat.fromSlice(flat[0..2]));

    try std.testing.expectEqual(Mat.zero().toArray(), Mat.init(0).toArray());
    try std.testing.expectEqual(Mat.one().toArray(), Mat.init(1).toArray());
}

test "identity and diagonal" {
    const Sq = MatrixF(f32, 3, 3);
    const identity = Sq.identity();
    try std.testing.expectEqual([3][3]f32{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    }, identity.toRows());

    const Rect = MatrixF(f32, 2, 3);
    const diag = Rect.fromDiagonal(.{ 4, 5 });
    try std.testing.expectEqual([2][3]f32{
        .{ 4, 0, 0 },
        .{ 0, 5, 0 },
    }, diag.toRows());
}

test "transpose" {
    const Mat = MatrixF(f32, 2, 3);
    const m = Mat.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    try std.testing.expectEqual([3][2]f32{
        .{ 1, 4 },
        .{ 2, 5 },
        .{ 3, 6 },
    }, m.transpose().toRows());
}

test "element-wise arithmetic" {
    const Mat = MatrixF(f32, 2, 2);
    const a = Mat.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = Mat.fromRows(.{
        .{ 10, 20 },
        .{ 30, 40 },
    });

    try std.testing.expectEqual([2][2]f32{
        .{ 11, 22 },
        .{ 33, 44 },
    }, a.add(b).toRows());
    try std.testing.expectEqual([2][2]f32{
        .{ -9, -18 },
        .{ -27, -36 },
    }, a.sub(b).toRows());
    try std.testing.expectEqual([2][2]f32{
        .{ -1, -2 },
        .{ -3, -4 },
    }, a.neg().toRows());
    try std.testing.expectEqual([2][2]f32{
        .{ 2, 4 },
        .{ 6, 8 },
    }, a.scale(2).toRows());
    try std.testing.expectEqual([2][2]f32{
        .{ 0.5, 1 },
        .{ 1.5, 2 },
    }, a.divScalar(2).toRows());
    try std.testing.expectEqual([2][2]f32{
        .{ 10, 40 },
        .{ 90, 160 },
    }, a.hadamard(b).toRows());
}

test "matrix multiplication and matrix-vector multiplication" {
    const A = MatrixF(f32, 2, 3);
    const B = MatrixF(f32, 3, 2);
    const a = A.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const b = B.fromRows(.{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });

    try std.testing.expectEqual([2][2]f32{
        .{ 58, 64 },
        .{ 139, 154 },
    }, a.mul(b).toRows());

    const x: @Vector(3, f32) = .{ 1, 2, 3 };
    try std.testing.expectEqual(@Vector(2, f32){ 14, 32 }, a.mulVec(x));
}

test "trace, determinant, and inverse" {
    const Mat = MatrixF(f64, 2, 2);
    const a = Mat.fromRows(.{
        .{ 4, 7 },
        .{ 2, 6 },
    });

    try std.testing.expectEqual(@as(f64, 10), a.trace());
    try std.testing.expectEqual(@as(f64, 10), a.determinant());

    const inv = a.inverse() orelse return error.TestUnexpectedResult;
    const expected_inv = Mat.fromRows(.{
        .{ 0.6, -0.7 },
        .{ -0.2, 0.4 },
    });
    try std.testing.expect(inv.approxEql(expected_inv, 1e-12));
    try std.testing.expect(a.mul(inv).approxEql(Mat.identity(), 1e-12));

    const Cube = MatrixF(f64, 3, 3);
    const c = Cube.fromRows(.{
        .{ 6, 1, 1 },
        .{ 4, -2, 5 },
        .{ 2, 8, 7 },
    });
    try std.testing.expectEqual(@as(f64, -306), c.determinant());
    try std.testing.expect(c.mul(c.inverse() orelse return error.TestUnexpectedResult).approxEql(Cube.identity(), 1e-10));

    const scalar = MatrixF(f64, 1, 1).fromArray(.{3});
    try std.testing.expectEqual(@as(f64, 3), scalar.determinant());
    try std.testing.expect(scalar.inverse().?.approxEql(MatrixF(f64, 1, 1).fromArray(.{1.0 / 3.0}), 1e-15));

    const singular = Mat.fromRows(.{
        .{ 1, 2 },
        .{ 2, 4 },
    });
    try std.testing.expectEqual(@as(f64, 0), singular.determinant());
    try std.testing.expect(singular.inverse() == null);
}

test "equality" {
    const Mat = MatrixF(f32, 2, 2);
    const a = Mat.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = Mat.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4.000001 },
    });

    try std.testing.expect(a.eql(a));
    try std.testing.expect(!a.eql(b));
    try std.testing.expect(a.approxEql(b, 0.001));
    try std.testing.expect(!a.approxEql(b, 0.0000001));
}

test "formatting" {
    const Mat = MatrixF(f32, 2, 2);
    const m = Mat.fromRows(.{
        .{ 1.5, 2.5 },
        .{ 3.5, 4.5 },
    });

    var buffer: [128]u8 = undefined;
    const printed = try std.fmt.bufPrint(&buffer, "{f}", .{m});
    try std.testing.expectEqualStrings(
        "[1.5, 2.5]\n[3.5, 4.5]",
        printed,
    );
}

test "diagonal, power, and norms" {
    const Mat = MatrixF(f64, 2, 2);
    var a = Mat.fromRows(.{
        .{ 3, 4 },
        .{ 0, 5 },
    });

    try std.testing.expectEqual([2]f64{ 3, 5 }, a.getDiagonal());
    a.setDiagonal(.{ 1, 2 });
    try std.testing.expectEqual([2][2]f64{
        .{ 1, 4 },
        .{ 0, 2 },
    }, a.toRows());

    const square = Mat.fromRows(.{
        .{ 3, 4 },
        .{ 0, 5 },
    });
    try std.testing.expectEqual([2][2]f64{
        .{ 9, 32 },
        .{ 0, 25 },
    }, square.pow(2).toRows());
    try std.testing.expect(square.pow(3).eql(square.mul(square).mul(square)));
    try std.testing.expect(square.pow(0).eql(Mat.identity()));

    try std.testing.expect(std.math.approxEqAbs(f64, @sqrt(@as(f64, 50)), square.frobeniusNorm(), 1e-12));
    try std.testing.expectEqual(@as(f64, 9), square.oneNorm());
    try std.testing.expectEqual(@as(f64, 7), square.infinityNorm());
    try std.testing.expectEqual(@as(f64, 5), square.maxNorm());
}

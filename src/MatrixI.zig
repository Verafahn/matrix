//! Integer matrices with statically known dimensions.

const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

/// Creates a row-major matrix with `row_cnt` rows and `col_cnt` columns whose
/// elements have the integer type `I`.
///
/// Arithmetic uses Zig's normal integer operators: overflow is a safety check
/// failure in Debug and ReleaseSafe, and is not checked in ReleaseFast or
/// ReleaseSmall.
///
/// ```zig
/// const Mat = MatrixI(i32, 2, 2);
/// const m = Mat.fromRows(.{ .{ 1, 2 }, .{ 3, 4 } });
/// ```
pub fn MatrixI(comptime I: type, comptime row_cnt: comptime_int, comptime col_cnt: comptime_int) type {
    comptime {
        if (@typeInfo(I) != .int) {
            @compileError("MatrixI requires an integer element type, got " ++ @typeName(I));
        }
    }
    return Matrix(I, row_cnt, col_cnt);
}

test "construction and arithmetic" {
    const Mat = MatrixI(i32, 2, 3);
    var m = Mat.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });

    try std.testing.expectEqual([2][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    }, m.toRows());
    try std.testing.expectEqual(@as(i32, 5), m.at(1, 1));

    m.set(0, 0, -1);
    try std.testing.expectEqual(@as(i32, -1), m.at(0, 0));

    const other = Mat.fromRows(.{
        .{ 10, 20, 30 },
        .{ 40, 50, 60 },
    });
    try std.testing.expectEqual([2][3]i32{
        .{ 9, 22, 33 },
        .{ 44, 55, 66 },
    }, m.add(other).toRows());
    try std.testing.expectEqual([2][3]i32{
        .{ -11, -18, -27 },
        .{ -36, -45, -54 },
    }, m.sub(other).toRows());
    try std.testing.expectEqual([2][3]i32{
        .{ -3, 6, 9 },
        .{ 12, 15, 18 },
    }, m.scale(3).toRows());
    try std.testing.expectEqual([2][3]i32{
        .{ -10, 40, 90 },
        .{ 160, 250, 360 },
    }, m.hadamard(other).toRows());
}

test "identity, diagonal, and transpose" {
    const Sq = MatrixI(i32, 3, 3);
    try std.testing.expectEqual([3][3]i32{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    }, Sq.identity().toRows());

    const Rect = MatrixI(i32, 2, 3);
    try std.testing.expectEqual([2][3]i32{
        .{ 7, 0, 0 },
        .{ 0, -3, 0 },
    }, Rect.fromDiagonal(.{ 7, -3 }).toRows());

    const m = Rect.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    try std.testing.expectEqual([3][2]i32{
        .{ 1, 4 },
        .{ 2, 5 },
        .{ 3, 6 },
    }, m.transpose().toRows());
}

test "matrix and matrix-vector multiplication" {
    const A = MatrixI(i32, 2, 3);
    const B = MatrixI(i32, 3, 2);
    const a = A.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const b = B.fromRows(.{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });

    try std.testing.expectEqual([2][2]i32{
        .{ 58, 64 },
        .{ 139, 154 },
    }, a.mul(b).toRows());

    const x: @Vector(3, i32) = .{ 1, 2, 3 };
    try std.testing.expectEqual(@Vector(2, i32){ 14, 32 }, a.mulVec(x));

    const identity = MatrixI(i32, 2, 2).identity();
    try std.testing.expectEqual([2][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    }, identity.mul(a).toRows());
}

test "equality, slices, and formatting" {
    const Mat = MatrixI(i32, 2, 2);
    const a = Mat.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const values = [_]i32{ 1, 2, 3, 4 };

    try std.testing.expect(a.eql(try Mat.fromSlice(&values)));
    try std.testing.expect(!a.eql(Mat.identity()));
    try std.testing.expectError(error.InvalidLength, Mat.fromSlice(values[0..1]));

    var buffer: [64]u8 = undefined;
    try std.testing.expectEqualStrings(
        "[1, 2]\n[3, 4]",
        try std.fmt.bufPrint(&buffer, "{f}", .{a}),
    );
}

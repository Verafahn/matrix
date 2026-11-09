const std = @import("std");
const matrix = @import("matrix");

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const Mat = matrix.MatrixF(f64, 2, 2);
    const a = Mat.fromRows(.{
        .{ 4, 7 },
        .{ 2, 6 },
    });
    const b = Mat.fromRows(.{
        .{ 1.5, 0 },
        .{ 0, 1.5 },
    });

    try stdout.print("A =\n{f}\n\n", .{a});
    try stdout.print("B =\n{f}\n\n", .{b});
    try stdout.print("A + B =\n{f}\n\n", .{a.add(b)});
    try stdout.print("A * B =\n{f}\n\n", .{a.mul(b)});
    try stdout.print("transpose(A) =\n{f}\n\n", .{a.transpose()});
    try stdout.print("trace(A) = {d}\n", .{a.trace()});
    try stdout.print("determinant(A) = {d}\n", .{a.determinant()});

    if (a.inverse()) |inverse| {
        try stdout.print("inverse(A) =\n{f}\n\n", .{inverse});
        try stdout.print("A * inverse(A) =\n{f}\n\n", .{a.mul(inverse)});
    }

    const x: @Vector(2, f64) = .{ 2, 3 };
    const y = a.mulVec(x);
    try stdout.print("A * {{ {d}, {d} }} = {{ {d}, {d} }}\n\n", .{ x[0], x[1], y[0], y[1] });

    const IntMat = matrix.MatrixI(i32, 2, 2);
    const int_a = IntMat.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const int_b = IntMat.identity().scale(3);
    try stdout.print("integer matrix:\n{f}\n\n", .{int_a.mul(int_b)});

    const vector: @Vector(2, i32) = .{ 5, 6 };
    const product = int_a.mulVec(vector);
    try stdout.print("integer A * {{ {d}, {d} }} = {{ {d}, {d} }}\n", .{ vector[0], vector[1], product[0], product[1] });

    try stdout.flush();
}

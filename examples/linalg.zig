const std = @import("std");
const matrix = @import("matrix");

const linalg = matrix.linalg;

fn printVector(comptime n: comptime_int, writer: *std.Io.Writer, values: @Vector(n, f64)) !void {
    const items: [n]f64 = values;
    try writer.writeAll("{ ");
    for (0..n) |i| {
        if (i != 0) try writer.writeAll(", ");
        try writer.print("{d:.4}", .{items[i]});
    }
    try writer.writeAll(" }");
}

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const Mat = matrix.MatrixF(f64, 3, 3);
    const a = Mat.fromRows(.{
        .{ 2, -1, 1 },
        .{ 1, 3, 2 },
        .{ 1, -1, 2 },
    });

    try stdout.writeAll("Linear solve\n");
    const expected: @Vector(3, f64) = .{ 1, 2, 3 };
    const b = a.mulVec(expected);
    if (linalg.solve(a, b)) |x| {
        try stdout.writeAll("x = ");
        try printVector(3, stdout, x);
        try stdout.writeByte('\n');
    } else {
        try stdout.writeAll("system is singular\n");
    }

    if (linalg.lu(a)) |decomposition| {
        try stdout.print("\nLU decomposition\ndeterminant = {d}\nL =\n{f}\nU =\n{f}\n", .{
            decomposition.determinant(),
            decomposition.lower,
            decomposition.upper,
        });
    }

    try stdout.writeAll("\nQR least squares\n");
    const Fit = matrix.MatrixF(f64, 4, 2);
    const samples = Fit.fromRows(.{
        .{ 1, 0 },
        .{ 1, 1 },
        .{ 1, 2 },
        .{ 1, 3 },
    });
    const observations: @Vector(4, f64) = .{ 1, 3, 5, 7 };
    if (linalg.leastSquares(samples, observations)) |coefficients| {
        try stdout.writeAll("fit = a + b*x, [a, b] = ");
        try printVector(2, stdout, coefficients);
        try stdout.writeByte('\n');
    }

    try stdout.writeAll("\nCholesky decomposition\n");
    const spd = Mat.fromRows(.{
        .{ 4, 12, -16 },
        .{ 12, 37, -43 },
        .{ -16, -43, 98 },
    });
    if (linalg.cholesky(spd)) |lower| {
        try stdout.print("L =\n{f}\n", .{lower});
    }

    try stdout.writeAll("\nSymmetric eigenvalues\n");
    const Sym = matrix.MatrixF(f64, 2, 2);
    const symmetric = Sym.fromRows(.{
        .{ 2, 1 },
        .{ 1, 2 },
    });
    if (linalg.eigenSymmetric(symmetric, 1e-12)) |eigen| {
        const values: [2]f64 = eigen.values;
        try stdout.print("eigenvalues = {{ {d:.4}, {d:.4} }}\n", .{ values[0], values[1] });
        try stdout.print("eigenvectors =\n{f}\n", .{eigen.vectors});
    }

    try stdout.writeAll("\nRank and null space\n");
    const dependent = Mat.fromRows(.{
        .{ 1, 2, 3 },
        .{ 2, 4, 6 },
        .{ 1, 1, 1 },
    });
    try stdout.print("rank = {d}\n", .{linalg.rank(dependent, 1e-10)});
    const null_space = linalg.nullSpace(dependent, 1e-10);
    for (null_space.basis()) |direction| {
        try stdout.writeAll("null direction = ");
        try printVector(3, stdout, direction);
        try stdout.writeByte('\n');
    }

    try stdout.flush();
}

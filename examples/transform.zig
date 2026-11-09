const std = @import("std");
const matrix = @import("matrix");

const transform = matrix.transform;

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const pi: f64 = std.math.pi;

    const translate = transform.translation2D(f64, 2, 1);
    const rotate = transform.rotation2D(f64, pi / 2);
    const model = rotate.mul(translate);

    const point: @Vector(3, f64) = .{ 1, 0, 1 };
    const transformed = model.mulVec(point);

    try stdout.writeAll("2D transform\n");
    try stdout.print("translation =\n{f}\n\n", .{translate});
    try stdout.print("rotation =\n{f}\n\n", .{rotate});
    try stdout.print("rotation * translation =\n{f}\n\n", .{model});
    try stdout.print(
        "(1, 0) -> ({d:.3}, {d:.3})\n\n",
        .{ transformed[0], transformed[1] },
    );

    const spin = transform.rotationAxis(f64, .{ 0, 1, 0 }, pi / 4);
    try stdout.writeAll("axis-angle rotation (45 degrees around Y):\n");
    try stdout.print("{f}\n\n", .{spin});

    const eye = @Vector(3, f64){ 2, 3, 4 };
    const view = transform.lookAt(f64, eye, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    const projection = transform.perspective(f64, pi / 3, @as(f64, 16) / @as(f64, 9), 0.1, 100);

    try stdout.writeAll("lookAt view matrix:\n");
    try stdout.print("{f}\n\n", .{view});

    try stdout.writeAll("perspective projection matrix:\n");
    try stdout.print("{f}\n", .{projection});

    try stdout.flush();
}

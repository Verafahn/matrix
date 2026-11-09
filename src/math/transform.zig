//! Common 2D and 3D affine transformation matrices.
//!
//! All functions return a matrix intended to be multiplied by a column vector
//! (`M * v`). Due to row-major storage, `a.mul(b)` applies `b` first and then
//! `a`. Homogeneous 2D transforms are 3x3 matrices and homogeneous 3D
//! transforms are 4x4 matrices.
//!
//! The projection helpers use a right-handed coordinate system looking down
//! negative Z and the depth range `[0, 1]` used by Vulkan and WebGPU.

const std = @import("std");
const MatrixF = @import("../MatrixF.zig").MatrixF;

const pi_f32: f32 = std.math.pi;

fn dot3(comptime F: type, a: @Vector(3, F), b: @Vector(3, F)) F {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn cross3(comptime F: type, a: @Vector(3, F), b: @Vector(3, F)) @Vector(3, F) {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn normalize3(comptime F: type, v: @Vector(3, F)) @Vector(3, F) {
    const length = @sqrt(dot3(F, v, v));
    std.debug.assert(length > 0);
    return v / @as(@Vector(3, F), @splat(length));
}

/// Returns a homogeneous 2D translation matrix.
pub fn translation2D(comptime F: type, x: F, y: F) MatrixF(F, 3, 3) {
    const Mat = MatrixF(F, 3, 3);
    return Mat.fromRows(.{
        .{ 1, 0, x },
        .{ 0, 1, y },
        .{ 0, 0, 1 },
    });
}

/// Returns a homogeneous 2D scaling matrix.
pub fn scaling2D(comptime F: type, x: F, y: F) MatrixF(F, 3, 3) {
    const Mat = MatrixF(F, 3, 3);
    return Mat.fromRows(.{
        .{ x, 0, 0 },
        .{ 0, y, 0 },
        .{ 0, 0, 1 },
    });
}

/// Returns a homogeneous 2D rotation matrix for `radians` around the origin.
pub fn rotation2D(comptime F: type, radians: F) MatrixF(F, 3, 3) {
    const Mat = MatrixF(F, 3, 3);
    const c = @cos(radians);
    const s = @sin(radians);
    return Mat.fromRows(.{
        .{ c, -s, 0 },
        .{ s, c, 0 },
        .{ 0, 0, 1 },
    });
}

/// Returns a homogeneous 3D translation matrix.
pub fn translation3D(comptime F: type, x: F, y: F, z: F) MatrixF(F, 4, 4) {
    const Mat = MatrixF(F, 4, 4);
    return Mat.fromRows(.{
        .{ 1, 0, 0, x },
        .{ 0, 1, 0, y },
        .{ 0, 0, 1, z },
        .{ 0, 0, 0, 1 },
    });
}

/// Returns a homogeneous 3D scaling matrix.
pub fn scaling3D(comptime F: type, x: F, y: F, z: F) MatrixF(F, 4, 4) {
    const Mat = MatrixF(F, 4, 4);
    return Mat.fromRows(.{
        .{ x, 0, 0, 0 },
        .{ 0, y, 0, 0 },
        .{ 0, 0, z, 0 },
        .{ 0, 0, 0, 1 },
    });
}

/// Returns a homogeneous 3D rotation matrix around the X axis.
pub fn rotationX(comptime F: type, radians: F) MatrixF(F, 4, 4) {
    const Mat = MatrixF(F, 4, 4);
    const c = @cos(radians);
    const s = @sin(radians);
    return Mat.fromRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, c, -s, 0 },
        .{ 0, s, c, 0 },
        .{ 0, 0, 0, 1 },
    });
}

/// Returns a homogeneous 3D rotation matrix around the Y axis.
pub fn rotationY(comptime F: type, radians: F) MatrixF(F, 4, 4) {
    const Mat = MatrixF(F, 4, 4);
    const c = @cos(radians);
    const s = @sin(radians);
    return Mat.fromRows(.{
        .{ c, 0, s, 0 },
        .{ 0, 1, 0, 0 },
        .{ -s, 0, c, 0 },
        .{ 0, 0, 0, 1 },
    });
}

/// Returns a homogeneous 3D rotation matrix around the Z axis.
pub fn rotationZ(comptime F: type, radians: F) MatrixF(F, 4, 4) {
    const Mat = MatrixF(F, 4, 4);
    const c = @cos(radians);
    const s = @sin(radians);
    return Mat.fromRows(.{
        .{ c, -s, 0, 0 },
        .{ s, c, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    });
}

/// Returns a homogeneous 3D rotation around an arbitrary axis.
///
/// `axis` does not need to be normalized, but it must not be zero.
pub fn rotationAxis(comptime F: type, axis: @Vector(3, F), radians: F) MatrixF(F, 4, 4) {
    const Mat = MatrixF(F, 4, 4);
    const a = normalize3(F, axis);
    const x = a[0];
    const y = a[1];
    const z = a[2];
    const c = @cos(radians);
    const s = @sin(radians);
    const t = 1 - c;

    return Mat.fromRows(.{
        .{ t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0 },
        .{ t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0 },
        .{ t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0 },
        .{ 0, 0, 0, 1 },
    });
}

/// Returns a right-handed look-at view matrix.
///
/// The camera is placed at `eye` and oriented towards `target`; `up` selects
/// the roll of the camera. `eye` must differ from `target`, and `up` must not
/// be parallel to the view direction.
pub fn lookAt(
    comptime F: type,
    eye: @Vector(3, F),
    target: @Vector(3, F),
    up: @Vector(3, F),
) MatrixF(F, 4, 4) {
    const Mat = MatrixF(F, 4, 4);
    const forward = normalize3(F, target - eye);
    const side = normalize3(F, cross3(F, forward, up));
    const upward = cross3(F, side, forward);

    return Mat.fromRows(.{
        .{ side[0], side[1], side[2], -dot3(F, side, eye) },
        .{ upward[0], upward[1], upward[2], -dot3(F, upward, eye) },
        .{ -forward[0], -forward[1], -forward[2], dot3(F, forward, eye) },
        .{ 0, 0, 0, 1 },
    });
}

/// Returns a right-handed perspective projection matrix with depth `[0, 1]`.
///
/// `fov_y_radians` is the vertical field of view, `aspect` is
/// `width / height`, and `near`/`far` are positive view-space distances.
pub fn perspective(
    comptime F: type,
    fov_y_radians: F,
    aspect: F,
    near: F,
    far: F,
) MatrixF(F, 4, 4) {
    std.debug.assert(fov_y_radians > 0 and fov_y_radians < std.math.pi);
    std.debug.assert(aspect > 0);
    std.debug.assert(near > 0 and far > near);

    const Mat = MatrixF(F, 4, 4);
    const f = 1 / @tan(fov_y_radians / 2);
    return Mat.fromRows(.{
        .{ f / aspect, 0, 0, 0 },
        .{ 0, f, 0, 0 },
        .{ 0, 0, far / (near - far), (far * near) / (near - far) },
        .{ 0, 0, -1, 0 },
    });
}

/// Returns a right-handed orthographic projection matrix with depth `[0, 1]`.
///
/// `left`/`right` and `bottom`/`top` define the view-space bounds; `near` and
/// `far` are positive view-space distances.
pub fn orthographic(
    comptime F: type,
    left: F,
    right: F,
    bottom: F,
    top: F,
    near: F,
    far: F,
) MatrixF(F, 4, 4) {
    std.debug.assert(right > left and top > bottom);
    std.debug.assert(near > 0 and far > near);

    const Mat = MatrixF(F, 4, 4);
    return Mat.fromRows(.{
        .{ 2 / (right - left), 0, 0, -(right + left) / (right - left) },
        .{ 0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom) },
        .{ 0, 0, 1 / (near - far), near / (near - far) },
        .{ 0, 0, 0, 1 },
    });
}

fn expectApproxVec(
    comptime n: comptime_int,
    expected: @Vector(n, f32),
    actual: @Vector(n, f32),
    tolerance: f32,
) !void {
    const e: [n]f32 = expected;
    const a: [n]f32 = actual;
    for (0..n) |i| {
        try std.testing.expect(std.math.approxEqAbs(f32, e[i], a[i], tolerance));
    }
}

test "2D translation, scaling, and rotation" {
    const translate = translation2D(f32, 3, -2);
    try expectApproxVec(3, .{ 4, 2, 1 }, translate.mulVec(.{ 1, 4, 1 }), 1e-6);

    const scale = scaling2D(f32, 2, 3);
    try expectApproxVec(3, .{ 2, 12, 1 }, scale.mulVec(.{ 1, 4, 1 }), 1e-6);

    const rotate = rotation2D(f32, pi_f32 / 2);
    try expectApproxVec(3, .{ 0, 2, 1 }, rotate.mulVec(.{ 2, 0, 1 }), 1e-6);

    // `a.mul(b)` applies `b` first: translate by (2, 0), then rotate 90°.
    const composed = rotate.mul(translation2D(f32, 2, 0));
    try expectApproxVec(3, .{ 0, 3, 1 }, composed.mulVec(.{ 1, 0, 1 }), 1e-6);
}

test "3D rotations" {
    const x = rotationX(f32, pi_f32 / 2);
    try expectApproxVec(4, .{ 0, 0, 1, 1 }, x.mulVec(.{ 0, 1, 0, 1 }), 1e-6);

    const y = rotationY(f32, pi_f32 / 2);
    try expectApproxVec(4, .{ 0, 0, -1, 1 }, y.mulVec(.{ 1, 0, 0, 1 }), 1e-6);

    const z = rotationZ(f32, pi_f32 / 2);
    try expectApproxVec(4, .{ 0, 1, 0, 1 }, z.mulVec(.{ 1, 0, 0, 1 }), 1e-6);

    const axis_z = rotationAxis(f32, .{ 0, 0, 5 }, pi_f32 / 2);
    try std.testing.expect(axis_z.approxEql(z, 1e-6));
}

test "3D translation and scaling" {
    const translate = translation3D(f32, 1, 2, 3);
    try expectApproxVec(4, .{ 2, 4, 6, 1 }, translate.mulVec(.{ 1, 2, 3, 1 }), 1e-6);

    const scale = scaling3D(f32, 2, 2, 2);
    try expectApproxVec(4, .{ 2, 4, 6, 1 }, scale.mulVec(.{ 1, 2, 3, 1 }), 1e-6);
}

test "lookAt places the eye at the origin" {
    const eye = @Vector(3, f32){ 2, 3, 4 };
    const view = lookAt(f32, eye, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    try expectApproxVec(4, .{ 0, 0, 0, 1 }, view.mulVec(.{ 2, 3, 4, 1 }), 1e-6);
}

test "perspective maps the near and far planes to zero and one" {
    const near: f32 = 0.1;
    const far: f32 = 100;
    const projection = perspective(f32, pi_f32 / 3, @as(f32, 16) / @as(f32, 9), near, far);

    const near_clip = projection.mulVec(.{ 0, 0, -near, 1 });
    try std.testing.expect(std.math.approxEqAbs(f32, 0, near_clip[2] / near_clip[3], 1e-6));

    const far_clip = projection.mulVec(.{ 0, 0, -far, 1 });
    try std.testing.expect(std.math.approxEqAbs(f32, 1, far_clip[2] / far_clip[3], 1e-6));
}

test "orthographic maps the near and far planes to zero and one" {
    const near: f32 = 0.5;
    const far: f32 = 50;
    const projection = orthographic(f32, -2, 2, -1, 1, near, far);

    const near_clip = projection.mulVec(.{ 0, 0, -near, 1 });
    try std.testing.expect(std.math.approxEqAbs(f32, 0, near_clip[2] / near_clip[3], 1e-6));

    const far_clip = projection.mulVec(.{ 0, 0, -far, 1 });
    try std.testing.expect(std.math.approxEqAbs(f32, 1, far_clip[2] / far_clip[3], 1e-6));
}

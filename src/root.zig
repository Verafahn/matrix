//! Static, row-major matrices for Zig.
//!
//! The module exposes two constructors:
//!
//! - `MatrixF(F, rows, cols)` for floating-point element types.
//! - `MatrixI(I, rows, cols)` for integer element types.
//!
//! Dimensions are compile-time constants, so shape mismatches are reported
//! while compiling. Common affine and projection transforms live in
//! `matrix.transform` (also available as `matrix.math.transform`).

pub const MatrixF = @import("MatrixF.zig").MatrixF;
pub const MatrixI = @import("MatrixI.zig").MatrixI;

/// 2D and 3D transformation matrices.
pub const transform = @import("math/transform.zig");

/// Math-related helpers.
pub const math = struct {
    pub const transform = @import("math/transform.zig");
};

test {
    _ = @import("MatrixF.zig");
    _ = @import("MatrixI.zig");
    _ = @import("math/transform.zig");
}

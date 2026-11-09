# Matrix

[![CI](https://github.com/Verafahn/matrix/actions/workflows/ci.yml/badge.svg)](https://github.com/Verafahn/matrix/actions/workflows/ci.yml)

Static, row-major matrices and common transformation math for Zig.

Matrix dimensions are compile-time constants, so incompatible shapes are
rejected by the compiler instead of producing runtime errors. Elements are
stored in a flat SIMD vector, and every operation has value semantics:
operations return a new matrix and never mutate their inputs.

## Features

- `MatrixF(F, rows, cols)` for `f16`, `f32`, `f64`, and other float types.
- `MatrixI(I, rows, cols)` for signed and unsigned integer types.
- Rectangular matrices, compile-time checked multiplication, and vector
  multiplication.
- Addition, subtraction, negation, scaling, Hadamard product, transpose, trace,
  determinant, and inverse.
- Exact and approximate equality, plus `{f}` formatting.
- 2D and 3D translation, scaling, rotation, look-at view, perspective, and
  orthographic projection helpers.
- No allocations and no dependencies beyond the Zig standard library.

## Requirements

- Zig 0.17.0 or newer.

## Installation

```sh
zig fetch --save git+https://github.com/Verafahn/matrix.git
```

Expose the module in `build.zig`:

```zig
const matrix = b.dependency("matrix", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("matrix", matrix.module("matrix"));
```

Then import it with `@import("matrix")`.

## Quick start

```zig
const matrix = @import("matrix");

const Mat = matrix.MatrixF(f32, 2, 2);

const a = Mat.fromRows(.{
    .{ 4, 7 },
    .{ 2, 6 },
});
const b = Mat.fromRows(.{
    .{ 1, 2 },
    .{ 3, 4 },
});

const sum = a.add(b);
const product = a.mul(b);
const transposed = a.transpose();
const x: @Vector(2, f32) = .{ 2, 3 };
const y = a.mulVec(x);

const det = a.determinant();
if (a.inverse()) |inverse| {
    // a.mul(inverse) is approximately the identity matrix.
    _ = inverse;
}
```

Printing uses the standard format method:

```zig
std.debug.print("product =\n{f}\n", .{product});
```

Integer matrices share the construction, access, and arithmetic API:

```zig
const IntMat = matrix.MatrixI(i32, 2, 3);
const m = IntMat.fromRows(.{
    .{ 1, 2, 3 },
    .{ 4, 5, 6 },
});

const identity = matrix.MatrixI(i32, 2, 2).identity();
const scaled = identity.scale(3);
const result = scaled.mul(m);
```

Integer arithmetic uses Zig's normal integer operators. Overflow is a safety
panic in Debug and ReleaseSafe and is not checked in ReleaseFast or
ReleaseSmall.

## Transformations

`matrix.transform` provides homogeneous transformation matrices for column
vectors (`M * v`). Because storage is row-major and matrices multiply on the
left, `a.mul(b)` applies `b` first and then `a`.

```zig
const transform = matrix.transform;

const pi: f64 = std.math.pi;

const translate = transform.translation2D(f64, 2, 1);
const rotate = transform.rotation2D(f64, pi / 2);
const model = rotate.mul(translate); // translate, then rotate

const point: @Vector(3, f64) = .{ 1, 0, 1 }; // homogeneous 2D point
const moved = model.mulVec(point);

const view = transform.lookAt(f64, .{ 2, 3, 4 }, .{ 0, 0, 0 }, .{ 0, 1, 0 });
const projection = transform.perspective(f64, pi / 3, @as(f64, 16) / @as(f64, 9), 0.1, 100);
```

Available helpers:

| Function | Result |
| --- | --- |
| `translation2D`, `scaling2D`, `rotation2D` | 3x3 homogeneous 2D matrices |
| `translation3D`, `scaling3D`, `rotationX`, `rotationY`, `rotationZ`, `rotationAxis` | 4x4 homogeneous 3D matrices |
| `lookAt` | 4x4 right-handed view matrix |
| `perspective`, `orthographic` | 4x4 right-handed projections with depth `[0, 1]` |

The projection helpers use the depth range used by Vulkan and WebGPU. The
camera looks down negative Z in view space.

## API overview

Both constructors return a matrix type with these compile-time constants:

| Constant | Meaning |
| --- | --- |
| `Ele` | Element type |
| `row`, `col` | Number of rows and columns |
| `rows`, `cols` | Aliases for `row` and `col` |
| `len` | Total number of elements |
| `is_square` | Whether `row == col` |

Constructors and conversion:

| Method | Description |
| --- | --- |
| `init(value)` | Matrix with every element equal to `value` |
| `zero()`, `one()` | Zero matrix or all-ones matrix |
| `identity()` | Identity matrix (square only) |
| `fromDiagonal(values)` | Diagonal matrix |
| `fromArray(array)`, `fromSlice(slice)` | Flat row-major input |
| `fromRows(rows)` | Nested array input |
| `toArray()`, `toRows()` | Convert back to arrays |
| `at(r, c)`, `set(r, c, value)` | Read or write one element |
| `getRow(r)`, `getCol(c)` | Read a row or column as a vector |

Operations:

| Method | Description |
| --- | --- |
| `add`, `sub`, `neg`, `scale`, `divScalar` | Element-wise arithmetic |
| `hadamard(other)` | Element-wise multiplication |
| `mul(other)` | Matrix multiplication |
| `mulVec(vector)` | Matrix-vector multiplication |
| `transpose()` | Transpose |
| `trace()` | Sum of the main diagonal (square only) |
| `determinant()` | Determinant (square float matrices only) |
| `inverse()` | Inverse or `null` if singular (square float matrices only) |
| `eql(other)` | Exact equality |
| `approxEql(other, tolerance)` | Approximate equality (float matrices only) |
| `format` | Supports `{f}` formatting |

Shape mismatches are compile errors. `fromSlice` reports `error.InvalidLength`
when its input has the wrong number of elements.

## Building and tests

```sh
zig build test            # unit tests
zig build examples        # build all examples
zig build run-examples    # run all examples
zig build run-basic       # run the basic example
zig build run-transform   # run the transform example
```

The examples live in `examples/` and are also built by CI.

## License

MIT. See [LICENSE](LICENSE).

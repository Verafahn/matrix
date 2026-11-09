# Matrix


## Requirements

- Zig 0.17.0 or newer.

## Installation

```sh
zig fetch --save git+https://github.com/Verafahn/matrix.git
```

Expose the module in `build.zig`:

```zig
const matrix = b.dependency("matrix", .{});
exe.root_module.addImport("matrix", matrix.module("matrix"));
```

Then import it with `@import("matrix")`.

## Quick start

## APIs

## Building and Tests

## License

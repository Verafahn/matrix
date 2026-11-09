const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("matrix", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_mod = b.addTest(.{ .root_module = mod });
    const run_test = b.addRunArtifact(test_mod);
    const test_step = b.step("test", "Test this module.");
    test_step.dependOn(&run_test.step);

    if(b.isRoot()) {
        // TODO build examples.
    }
}

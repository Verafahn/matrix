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

    const examples_step = b.step("examples", "Build all examples.");
    const run_examples_step = b.step("run-examples", "Run all examples.");

    if (b.isRoot()) {
        const example_names = [_][]const u8{ "basic", "transform", "linalg" };
        for (example_names) |name| {
            const exe = b.addExecutable(.{
                .name = name,
                .root_module = b.createModule(.{
                    .root_source_file = b.path(b.fmt("examples/{s}.zig", .{name})),
                    .target = target,
                    .optimize = optimize,
                }),
            });
            exe.root_module.addImport("matrix", mod);
            examples_step.dependOn(&exe.step);

            const run = b.addRunArtifact(exe);
            run_examples_step.dependOn(&run.step);

            const run_step = b.step(
                b.fmt("run-{s}", .{name}),
                b.fmt("Run the {s} example.", .{name}),
            );
            run_step.dependOn(&run.step);
        }
    }
}

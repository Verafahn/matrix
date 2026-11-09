pub fn MatrixF(comptime F: type, comptime row_cnt: comptime_int, comptime col_cnt: comptime_int) type {
    comptime {
        if(@typeInfo(F) != .float) {
            @compileError("");
        }
    }
    return struct {
        const Self = @This();
        pub const Ele = F;
        pub const row = row_cnt;
        pub const col = col_cnt;

        data: @Vector(row * col, Ele),

        pub fn init() Self {
            
        }
    };
}

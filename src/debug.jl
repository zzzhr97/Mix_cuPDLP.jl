# debug.jl

is_debug = false

mutable struct DebugPrinter
    cnt::Int
    list::Array{Any, 1}
end

global debug = DebugPrinter(0, Int64[])

function debug_reset!()
    global debug
    debug.cnt = 0
    debug.list = Any[]
end

function debug_add!(value::Any=NaN)
    global debug
    debug.cnt += 1
    if isnan(value) == false
        debug.list = push!(debug.list, value)
    end
end

function debug_print(str1::String, str2::String="List: ")
    if is_debug == false
        return
    end

    global debug
    println(str1, debug.cnt)
    if length(debug.list) > 0
        println(debug.list)
    end
end

function debug_test()
    debug_add!(2.0)
    debug_add!(3)
    debug_print("Hello, debug_test: ")
end
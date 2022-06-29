using CUDA

function say(num)
    @cuprintf("Thread %ld says: %ld\n",threadIdx().x,num)
    return
end

function type()
    tp = typeof(threadIdx().x)
    @cuprintln("$(tp == Int64)")
    return
end

#@cuda threads = 8 say(42)

@cuda threads=1 type()
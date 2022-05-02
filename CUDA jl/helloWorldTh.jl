using CUDA

function say(num)
    @cuprintf("Thread %ld says: %ld\n",threadIdx().x,num)
    return
end

@cuda threads = 8 say(42)
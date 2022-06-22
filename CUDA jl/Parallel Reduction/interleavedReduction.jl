using CUDA

function interleaved_reduction_kernel(g_out_g_in, size)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    s_data = CUDA.CuStaticSharedArray(Float32, size)

    s_data[threadIdx().x] = idx < size ? g_in[idx] : 0.f0
    CUDA.sync_threads()

    #reducción
    #Interleaved addressing

    stride = 1

    while stride < blockDim().x 

        index = 2 * stride * threadIdx().x 
        if index < blockDim().x
            s_data[index] += s_data[index + stride]
        end

        CUDA.sync_threads()

        global stride *= 2
    end

    if threadIdx().x == 1
        g_out[blockIdx().x] = s_data[1]
    end

    return nothing
end

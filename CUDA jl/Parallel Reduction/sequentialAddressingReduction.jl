using CUDA 

function sequential_reduction_kernel(g_out, g_in, size)
    idx = (blockIdx().x -1) * blockDim().x + threadIdx().x 

    s_data = CUDA.CuStaticSharedArray(Float32,size)

    s_data[threadIdx().x] = (idx < size) ? g_in[idx] : 0.f0

    CUDA.sync_threads()

    #Reduccion
    #Sequential Addressing 
    stride = div(blockDim().x,2)
    while stride > 0

        if threadIdx().x < stride
            s_data[threadIdx().x] += s_data[threadIdx().x + stride]
        end
        CUDA.sync_threads()

        stride >>= 1
    end

    if threadIdx().x == 1
        g_out[blockIdx().x] = s_data[1]
    end

    return nothing
end
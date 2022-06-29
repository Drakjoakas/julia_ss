using CUDA

#Algoritmo de reducción aplicado de forma "ingenuea"

function naive_reduction_kernel(data_out, data_in, stride, size)
    idx = (blockIdx().x -1) * blockDim().x + threadIdx().x 
    if (idx + stride < size)
        data_out[idx] += data_in[idx + stride]
    end
    return nothing
end

function naive_reduction(d_out, d_in, n_threads, size)
    n_blocks = div((size + n_threads -1) , n_threads)

    global stride = 1
    while stride < size
        tmp = stride
        @cuda threads=n_threads blocks=n_blocks naive_reduction_kernel(d_out, d_in, tmp, size)
        global stride *= 2
    end
end


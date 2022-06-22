using CUDA

#Algoritmo de reducción aplicado de forma "ingenuea"

function naive_reduction_kernel(data_out, data_in, stride, size)
    idx = (blockIdx().x -1) * blockDim().x + threadIdx().x 
    if (idx + stride < size)
        data_out[idx] += data_in[idx + stride]
    end
end

function naive_reduction(d_out, d_in, n_threads, size)
    n_blocks = div((size + n_threads -1) , n_threads)

    stride = 1
    while stride < size
        @cuda threads=n_threads blocks=n_blocks naive_reduction_kernel(d_out, d_in, stride, size)
        global size *= 2
    end
end


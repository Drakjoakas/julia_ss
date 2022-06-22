using CUDA

function reduction_kernel(d_out, d_in, size)
    idx = (blockIdx().x -1) * blockDim().x + threadIdx().x 

    s_data = CuStaticSharedArray(Float32, size)

    s_data[threadIdx().x] = (idx < size) ? d_in[idx] : 0.0f0

    CUDA.sync_threads()

    #Reducion 
    stride = 1
    while stride < blockDim().x
        #Mejora en rendimiento: usar operaciones de bits
        #if (idx & (stride * 2 - 1)) == 0
        if idx % (stride *2) == 0
            s_data[threadIdx().x] += s_data[threadIdx().x + stride]
        end
        CUDA.sync_threads()
        global stride *= 2
    end

    if threadIdx().x == 1
        d_out[blockIdx().x] = s_data[1]
    end
    return nothing
end

function reduction( d_out, d_in, n_threads, size)
    d_out = d_in
    while size > 1
        n_blocks = div(size + n_threads-1, n_threads)
        # TODO
        # lanzamiento en CUDA C:
        # reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>> (d_out,d_in,size)
        # ¿Qué son el tercer y cuarto parámetro?
        @cuda threads=n_blocks blokcs=n_blocks reduction_kernel(d_out, d_in, size)
        global size = n_blocks
    end
end

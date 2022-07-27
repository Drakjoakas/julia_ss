using CUDA

function reduction_kernel(d_out, d_in, size::Integer, threadsPerBlock)
    idx::Int64 = (blockIdx().x -1) * blockDim().x + threadIdx().x 

    s_data = CUDA.CuDynamicSharedArray(Float64,threadsPerBlock*2)

    s_data[threadIdx().x] = (idx < size) ? d_in[idx] : 0.0f0

    CUDA.sync_threads()

    #Reduccion 
    #stride::Int64 = 1
    #while stride < blockDim().x
    for stride = 1:(blockDim().x-1)

        if log(2,stride) % 1 != 0
            continue
        end
        
        #Mejora en rendimiento: usar operaciones de bits
        #if (idx & (stride * 2 - 1)) == 0
        if (idx % (stride * 2)) == 0
            #tmp::Float64 = s_data[threadIdx().x] + s_data[threadIdx().x + stride]
            @cuprintln("[$(idx)] = $(data_out[idx]) + [$idx + $stride]$(data_out[idx+stride])")
            s_data[threadIdx().x] += s_data[threadIdx().x + stride]
        end
        CUDA.sync_threads()
        
        #newStride = stride * 2
        #global stride =  newStride

    end
    
    if threadIdx().x == 1
        d_out[blockIdx().x] = s_data[1]
    end
    return nothing
end

function reduction_kernel_grid_strided_loop(d_out, d_in, size)
    idx = (blockIdx().x -1) * blockDim().x + threadIdx().x 

    s_data = CuStaticSharedArray(Float32, size)

    #cumulates input with grid-stride loop  and save to the shared memory
    input = 0.f0
    for i = idx:(blockDim().x * gridDim().x):size
        input += d_in[i]
    end
    s_data[threadIdx().x] = input
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

    while size > 1
        
        n_blocks = div(size + n_threads-1, n_threads)
        # lanzamiento en CUDA C:
        # reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>> (d_out,d_in,size)
        # ¿Qué son el tercer y cuarto parámetro?
        println("Num Blocks: $n_blocks")
        memSize = n_threads* sizeof(Float64) * 2
        @cuda threads=n_threads blocks=n_blocks shmem=memSize #=stream=CuStream()=# reduction_kernel(d_out, d_in, size,n_threads)
        #println("#"^64)
        println(Array(d_out))
        size = n_blocks
        
    end
end

function reduction_w_grid_size(d_out, d_in, n_threads, size)
    num_sms = 0 #getMultiProcessorCount
    num_blocks_per_sm = 0 #cudaOccupancyMaxActiveBlocksPerMultiprocessor

    n_blocks = min(num_blocks_per_sm * num_sms, Int((size + n_threads+1)/n_threads))
    # TODO
    ##Lanzar kernel
    #reduction_kernel_grid_strided_loop<<<n_blocks,n_threads,n_threads*sizeof(float),0>>>(params,size)
    #reduction_kernel_grid_strided_loop<<<1,n_threads,n_threads*sizeof(float),0>>>(params,n_blocks)

end

A = zeros(100)
B = similar(A)

for i = eachindex(A)
    A[i] = i
end

println(A)

d_a = CuArray(A)
d_b = CuArray(B)

#naive_reduction(d_b, d_a, 32, length(A))
reduction(d_b,d_a,32,length(A))

B = Array(d_b)

println(B)
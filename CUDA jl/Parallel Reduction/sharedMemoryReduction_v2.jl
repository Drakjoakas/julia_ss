using CUDA


function shmem_reduce_kernel(d_out, d_in)

    s_data = CUDA.CuDynamicSharedArray(Float64,blockDim().x)

    myId = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    tid  = threadIdx().x 

    #Load shared mem from global mem 
    s_data[tid] = d_in[myId]
    CUDA.sync_threads()  #Make sure the entire block is loaded

    #do reduction in shared mem 
    st = div(blockDim().x , 2)
    for s = st:-1:1
        if s != st
            continue
        end

        if tid < s
            s_data[tid] += s_data[tid + s]
        end
        CUDA.sync_threads()
        global st >>= 1
    end

    if tid == 1
        d_out[blockIdx().x] = s_data[1]
    end

    return nothing
end

function reduce(d_out, d_in, size)
    th = 32
    bl = size / th

    @cuda threads=th blocks=bl shmem=(th*sizeof(Float64)) shmem_reduce_kernel(d_out, d_in)
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
reduce(d_b,d_a,32)

B = Array(d_b)

println(B)

#=
st = 100
for i = st:-1:0
    if i != st
        continue
    end
    println(i)
    global st >>= 1
end
#==#
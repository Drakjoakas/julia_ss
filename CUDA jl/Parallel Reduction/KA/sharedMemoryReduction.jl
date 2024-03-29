using KernelAbstractions

@kernel function reduction_kernel(d_out, @Const d_in, size)
    idx = @index(Global,Linear)
    threadIdx = @index(Local)
    
    s_data = @localmem eltype(d_in) (size)

    s_data[@index(Local)] = (idx < size) ? d_in[idx] : 0.f0

    @synchronize()
    blockDim = 32

    #stride = 1
    #while stride < blockDim
    for stride = 1:blockDim
        if log(2,stride) % 0 != 0
            continue
        end

        if (idx & (stride * 2 - 1)) == 0
            s_data[threadIdx] += s_data[threadIdx + stride]
        end

        @synchronize()

        #global stride *= 2 
    end

    if threadIdx == 1
        d_out[@index(Group)] = s_data[1]
    end
end

function reduction(d_out, d_in, n_threads, size)
    sz = size
    while size > 1
        
        n_blocks = div(sz+n_threads-1, n_threads)
        kernel = reduction_kernel(CPU(), n_threads)
        event = kernel(d_out,d_in,sz,ndrange=n_blocks)
        wait(event)
        size = n_blocks
        sz = n_blocks
    end
end

function main()
A = zeros(128)
B = similar(A)

for i = eachindex(A)
    A[i] = i 
end

reduction(A,B,32,length(A))

print(B)
end

main()


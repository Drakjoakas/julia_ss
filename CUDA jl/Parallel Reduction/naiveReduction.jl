using CUDA

#Algoritmo de reducción aplicado de forma "ingenuea"

function naive_reduction_kernel(data_out, data_in, stride, size)
    idx = (blockIdx().x -1) * blockDim().x + threadIdx().x 
    if (idx + stride <= size)
        #@cuprintln("ID = $idx; stride = $stride")
        #@cuprintln("[$idx] = $(data_out[idx]) + [$idx + $stride]$(data_out[idx+stride])")
        data_out[idx] += data_in[idx + stride]
    end
    return nothing
end

function naive_reduction(d_out, d_in, n_threads, size)
    n_blocks = div((size + n_threads -1) , n_threads)
    
    #while stride < size
    for stride = 1:size
        if log(2,stride) % 1 != 0
            continue
        end

        @cuda threads=n_threads blocks=n_blocks naive_reduction_kernel(d_out, d_in, stride, size)
        #global stride *= 2
        
    end
end


using KernelAbstractions, CUDA, CUDAKernels, Test

@kernel function matMul!(a, b, c)
    i,j = @index(Global, NTuple)
    
    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        tmp_sum += a[i,k] * b[k,j]
    end

    c[i,j] = tmp_sum

end

#Kernel wrapper to avoid errors when launching
function matmul!(a,b,c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, Array)
        kernel! = matMul!(CPU(),4)
    else
        kernel! = matMul!(CUDADevice(),256)
    end
    kernel!(a,b,c,ndrange=size(c))
end

a = rand(256,123)
b = rand(123, 45)
c = zeros(256,45)

if has_cuda_gpu()
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)

    ev = matmul!(d_a, d_b, d_c)
    wait(ev)

end
@test isapprox(Array(d_c), a*b)


using CUDA, Test

const M = 256
const N = 256
const Ñ = 256 

function matmul_kernel_cu!(a,b,c)
    i = threadIdx().x + (blockIdx().x -1) * blockDim().x
    j = threadIdx().y + (blockIdx().y -1) * blockDim().y 

    tmp_sum = zero(eltype(c))

    for k = 1:size(a)[2]
        tmp_sum += a[i,k] * b[k,j]
    end

    c[i,j] = tmp_sum
    return nothing
end

function matmul_cu(a,b,c)
    if size(a)[2] != size(b)[1]
        println("El tamaño de las matrices no coincide.")
        return nothing
    end

    m,n = size(c)
    n_threads = (32,32)
    n_blocks = (div(m,32),div(m,32))
    println("$m , $n , $n_threads, $n_blocks")
    @sync @cuda threads=n_threads blocks=n_blocks matmul_kernel_cu!(a,b,c)

end

d_a = CUDA.rand(M,N)
d_b = CUDA.rand(N,Ñ)
d_c = CuArray(zeros(M,Ñ))

matmul_cu(d_a,d_b,d_c)

@test isapprox(Array(d_c), Array(d_a)*Array(d_b))
using CUDA

const BLOCK_DIM = 16


# Programa para revisar la ocupación de CUDA (CUDA Occupancy)

function segmm_gpu_kernel(A,B,C,N::Int,M::Int,K::Int,alpha,beta)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    sum = 0.0

    for i in 0:K
        sum += A[row*K+i] * B[i*K+col]
    end
    C[row * M + col] = alpha * sum + beta *c[row * M + col]
    return nothing
end

function segmm_gpu(A, B, C, N, M, K, alpha, beta)
    blockDim = (BLOCK_DIM,BLOCK_DIM)
    gridDim = (div(M,blockDim[1]),div(N,blockDim[2]))
    @cuda threads=blockDim blocks=gridDim segmm_gpu_kernel(A,B,C,N,M,K;alpha,beta)
end

function main()
    alpha = 2.0
    beta  = 1.0
    N = M = K = 2048

    A = rand(N,K)
    B = rand(K,M)
    C = rand(N,M)

    d_A = CuArray(A)
    d_B = CuArray(B)
    d_C = CuArray(C)

    segmm_gpu(d_A, d_B, d_C, N, M, K, alpha, beta)

    C = Array(d_C)

    d_A = nothing
    d_B = nothing
    d_C = nothing
end

#TODO
#Probar qué es lo que regresa launch_configuration()
#¿Sirve para un kernel bidimensional?
#Revisar con el profiler de NVIDIA
function segmm_gpu_bounded(A,B,C, N, M, K, alpha, beta)
    kernel  = @cuda launch=false segmm_gpu_kernel(A,B,C,N,M,K, alpha, beta)
    config  = launch_configuration(kernel.fun)
    threads = min(N,config.threads)
    blocks  = cld(N,threads)

    CUDA.@sync begin
        kernel(A,B,C,N,M,K, alpha, beta; threads, blocks)
    end
end

main()


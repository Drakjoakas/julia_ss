using CUDA

const global N = 64
const global BLOC_SIZE = 32

function showArray(a::AbstractArray) 
    cont = 0
    for ix in eachindex(a)
        print(a[ix] +" ")
        cont++
        if cont == N
            println("")
        end
    end
end

function matrix_transpose_naive!(input, output)
    indexX::Int = threadIdx().x + blockIdx().x*blockDim().x
    indexY::Int = threadIdx().y + blockIdx().y*blockDim().y
    index::Int = indexY * N +indexX
    transposedIndex::Int = indexX * N + indexY

    output[transposedIndex] = input[index]
end

function matrix_transpose_shared!(input, output)
    sharedMemory = CUDA.CuStaticSharedArray(Float64,(32 + 1,32))

    #global index
    indexX = threadIdx().x + (blockIdx().x-1) * blockDim().x
    indexY = threadIdx().y + (blockIdx().y-1) * blockDim().y 

    #transposed global index
    tindexX = threadIdx().x + (blockIdx().y-1) * blockDim().x 
    tindexY = threadIdx().y + (blockIdx().x-1) * blockDim().y 

    #local index 
    localIndexX = threadIdx().x 
    localIndexY = threadIdx().y 
    index = (indexY-1) * 64 + indexX 
    transposedIndex = (tindexY-1) * 64 + tindexX

    #transposed the matrix in shared memory 
    #global memory is read in coalesced fashion

    sharedMemory[localIndexX,localIndexY] = input[index]
    ##syncthreads() ?
    CUDA.sync_threads()
    output[transposedIndex] = sharedMemory[localIndexY,localIndexX]
    return 
end

size  = N * N 

a = rand(64,64)
b = similar(a)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(b)

blockSize = (BLOC_SIZE,BLOC_SIZE,1)
gridSize  = (Int(N/BLOC_SIZE),Int(N/BLOC_SIZE),1)

#@cuda threads=blockSize blocks=gridSize matrix_transpose_naive!(da,db)

@cuda threads=blockSize blocks=gridSize matrix_transpose_shared!(d_a,d_c)

b=Array(d_b)
c=Array(d_c)

@assert a' == c

#Liberamos la memoria
d_a = nothing
d_b = nothing
d_c = nothing

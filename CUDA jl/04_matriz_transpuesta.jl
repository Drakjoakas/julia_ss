using CUDA

const global N = 1024
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
    sharedMemory = CuStaticSharedArray(Int,(BLOC_SIZE,BLOC_SIZE+1))

    #Global Index 
    indexX::Int = threadIdx().x + blockIdx().x * blockDim().x
    indexY::Int = threadIdx().y + blockIdx().y * blockDim().y

    #transposed global memory index 
    tindexX::Int = threadIdx().x + blockIdx().y * blockDim().x
    tindexY::Int = threadIdx().y + blockIdx().x * blockDim().y

    #local index
    localIndexX::Int = threadIdx().x
    localIndexY::Int = threadIdx().y

    #
    index = indexY * N + indexX
    transposedIndex = tindexY * N + tindexX

    #leyendo de memoria global de manera fusionada y realizando la transposición en memoria compartida
    sharedMemory[localIndexX,localIndexY] = input[index]
    
    sync_threads()

    output[transposedIndex] = sharedMemory[localIndexY,localIndexX]

end

size  = N * N 

a = Array{Int32,size}
b = Array{Int32,size}
for i in eachindex(a)
    a[i] = i
end

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(b)

blockSize = (BLOC_SIZE,BLOC_SIZE+1)
gridSize  = (N/BLOC_SIZE,N/BLOC_SIZE,1)

@cuda threads=blockSize blocks=gridSize matrix_transpose_naive!(da,db)

@cuda threads=blockSize blocks=gridSize matrix_transpose_shared!(da,dc)

b=Array(d_b)
c=Array(d_c)

#Liberamos la memoria
d_a = nothing
d_b = nothing
d_c = nothing

using CUDA

const N = 64
const TILE_DIM = 32
const BLOCK_ROWS = 8

function matriz_transpose_cd_nv!(A, B)
    indexX = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
    indexY = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    for j in 0:blockDim().y:blockDim().x-1
        A[indexX,indexY+j] = B[indexY + j, indexX]
    end

    return nothing
end


dimGrid  = (Int(N/TILE_DIM), Int(N/TILE_DIM),1)
dimBlock = (TILE_DIM,BLOCK_ROWS,1)

B  = CUDA.rand(N,N)
A  = similar(B)



@cuda threads=dimBlock blocks=dimGrid matriz_transpose_cd_nv!(A,B)


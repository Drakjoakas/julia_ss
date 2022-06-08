using KernelAbstractions
import KernelAbstractions.Extras: @unroll

@kernel function transpose_coalesced(A, @Const(B))
    TILE_DIM   = @uniform groupsize()[1]
    BLOCK_ROWS = @uniform groupsize()[2]

    tile = @localmem eltype(A) (TILE_DIM+1,TILE_DIM)

    i, j  = @index(Local,NTuple)
    gi,gj = @index(Group,NTuple)

    I = (gi-1) * TILE_DIM + i #No podemos usar @index(Global)
    J = (gj-1) * TILE_DIM + j #Porqie el rango nd está comprimido

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds tile[i,j+k] = B[I,J+k]
    end

    @synchronize

    I = (gj-1) * TILE_DIM + i
    J = (gi-1) * TILE_DIM + j

    @unroll for K in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds A[I, J+k] = tile[j+k,i]
    end
end

using CUDA

function ka_transpose(
    A::CuMatrix, 
    B::CuMatrix, 
    ::Val{TILE_DIM}=Val(32),
    ::Val{BLOCK_ROWS}=Val(8);
    dependencies=nothing
    ) where {TILE_DIM, BLOCK_ROWS}

    @assert TILE_DIM % BLOCK_ROWS == 0
    @assert size(A) == size(B) && size(A,1) == size(A,2)
    @assert size(A,1) % TILE_DIM == 0

    block_factor = div(TILE_DIM,BLOCK_ROWS)
    ndrange = (size(A,1), div(size(A,2), block_factor))

    kernel = transpose_coalesced(CUDADevice(), (TILE_DIM, BLOCK_ROWS))
    event  = kernel(A,B, ndrange= ndrange, dependencies = dependencies)
    return event
end


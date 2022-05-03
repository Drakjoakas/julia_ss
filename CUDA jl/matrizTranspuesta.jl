using CUDA

COLS  = 10
FILAS = 6

function showArray(array::Array, cols::Int, row::Int)
    for i in 1:row
        for j in 1:cols
            print("$(array[i + (j-1)*row]) " )
        end
        println(" ")
    end
end


function matrizTranspuesta!(entrada,salida)
    columna  = threadIdx().x
    fila     = threadIdx().y
    univId   = (columna) + (fila-1) * blockDim().x
    transId  = (fila) + (columna-1) * blockDim().y
    # @cuprintln("Idx = $(threadIdx().x), Idy=$(threadIdx().y) Id1= $univId, Id2=$transId, blockDim=$(blockDim().x)")
    salida[transId] = entrada[univId]
    
    return nothing 
end

entrada_hst = zeros(Int32,FILAS,COLS)
for i in eachindex(entrada_hst)
    entrada_hst[i] = i
end

entrada_dev = CuArray(entrada_hst)
salida_dev  = CuArray(zeros(Int64,COLS,FILAS))

@sync @cuda threads=(FILAS,COLS) matrizTranspuesta!(entrada_dev,salida_dev)

salida_hst = Array(salida_dev)

println("ARREGLO DE ENTRADA: ")
showArray(entrada_hst,COLS,FILAS)

println("ARREGLO DE SALIDA: ")
showArray(salida_hst,FILAS,COLS)



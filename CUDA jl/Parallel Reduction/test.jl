include("naiveReduction.jl")
#include("sharedMemoryReduction.jl")

const size = 100

A = zeros(size)


for i = eachindex(A)
    A[i] = i
end

d_a = CuArray(A)
d_b = d_a
println(A)
naive_reduction(d_b, d_a, 32, length(A))
#reduction(d_b,d_a,32,length(A))

B = Array(d_b)

println("Sum from 1 to $size: $(B[1])")
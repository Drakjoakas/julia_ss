include("naiveReduction.jl")
include("sharedMemoryReduction.jl")

A = zeros(128)
B = similar(A)

for i = eachindex(A)
    A[i] = i
end

d_a = CuArray(A)
d_b = CuArray(B)

#naive_reduction(d_b, d_a, 32, length(A))
reduction(d_b,d_a,32,length(A))

B = Array(d_b)

println(B)
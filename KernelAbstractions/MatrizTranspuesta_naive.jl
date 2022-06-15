using KernelAbstractions, CUDA, CUDAKernels, Test

@kernel function transpose_kernel!(A, @Const B )
    I, J = @index(Global, NTuple)
    @inbounds A[I,J] = B[J,I]
end

B = CUDA.rand(1024,1024)
A = similar(B)

const transpose! = transpose_kernel!(CUDADevice(),(32,32))
event = transpose!(A,B, ndrange=size(A))
wait(event)
@show A
@show B 
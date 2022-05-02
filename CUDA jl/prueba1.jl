using CUDA

sigmoid(x) = 1/(1+exp(-x))

a = CuArray([1.])
b = CuArray([1.])
c = CuArray([1.])
d = CuArray([1.])
function elwise_kernel(op,a)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    a[i] = op(a[i])
    return
end

@cuda elwise_kernel(sigmoid,a)

map(sigmoid,b)
sigmoid.(c)

1 ./ (1 .+ exp.(-d))
print(a)
print(b)
print(c)
print(d)
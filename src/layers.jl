using Flux, NNlib

function fixdensekernel(W::AbstractArray{T,2}) where T <: Number
    WW = permutedims(W,(2,1))
    return Float32.(WW)
end

function fixconvkernel(W::AbstractArray{T,4}) where T <: Number
    WW = permutedims(W,(2,1,3,4))
    WW = NNlib.flipweight(WW)
    return Float32.(WW)
end

function fixbias(b)
    return Float32.(b)
end

function FConv(W, b, σ = identity; pad = 0, 
    stride = 1, dilation = 1)
    return Conv(fixconvkernel(W),fixbias(b), σ; 
        pad = pad, stride = stride, dilation = dilation)
end

function FDense(W,b) Dense(fixdensekernel(W), fixbias(b)) end

function Fflatten(x::AbstractArray{T, 4}) where T <: Number
    return reshape(permutedims(x,(2,1,3,4)),:,size(x)[end])
end

function Fflatten(x::AbstractArray{T, 2}) where T <: Number
    return reshape(permutedims(x,(2,1)),:,size(x)[end])
end
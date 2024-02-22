function envelope(X, Y, ξ)
    ξ = collect(ξ)
    logdξ = log(ξ[2]) - log(ξ[1])
    bins = [Float64[] for n = 1:length(ξ)]
    dmin = logdξ * ones(length(ξ))
    ymin = zeros(length(ξ))
    for (x, y) in zip(X, Y)
       d, i = findmin(abs.(ξ .- x))
       logd = abs(log(ξ[i]) - log(x))
       if  logd < 0.25 * logdξ
           push!(bins[i], y)
       else
           if logd < dmin[i]
               dmin[i] = d
               ymin[i] = y
           end
       end
    end
    for i = 1:length(bins)
        push!(bins[i], ymin[i])
    end
    I = findall( length.(bins) .> 0 )
    bins = bins[I]
    ξ = ξ[I]
    μ = zeros(length(ξ))
    σ = zeros(length(ξ))
    mx = maximum.(bins)
    return ξ, mx
 end
 
 function sparsify(x, y; dist=:loglog, tol = 0.05)
     if dist == :loglog
         d = (z1, z2) -> norm(log.(z1) - log.(z2))
     else
         error("unknown dist")
     end
     xnew = Float64[x[1]]
     ynew = Float64[y[1]]
     for (s, t) in zip(x,  y)
         minδ = minimum( d([s,t], [sn, tn]) for (sn, tn) in zip(xnew, ynew) )
         if minδ > tol
             push!(xnew, s)
             push!(ynew, t)
         end
     end
     return xnew, ynew
 end
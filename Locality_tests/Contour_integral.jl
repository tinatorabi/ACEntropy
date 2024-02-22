using ComplexElliptic
import ComplexElliptic.ellipkkp
import ComplexElliptic.ellipjc
using LinearAlgebra

function Contour_integral(A,f)
    e =  eigen(A).values
    m = e[argmin(abs.(e))]
    M = e[argmax(abs.(e))]
    k = (Complex(M/m)^(1/4) - 1) / (Complex(M/m)^(1/4) + 1)
    L = -log(k) / π
    KK, Kp = ellipkkp(L)
    S = zeros(Complex{Float64}, size(A))
    for N in 5:5:15
        t = 0.5im * Kp .- KK .+ (0.5:N) .* (2 * KK / N)
        u, c, d = ellipjc(t, L)
        w = Complex(m * M)^(1/4) .* ((1/k .+ u) ./ (1/k .- u))
        dzdt = c .* d ./ (1/k .- u).^2
        S = zeros(Complex{Float64}, size(A))
        for j in 1:N
            S = S + (f(w[j]^2) / w[j]) * inv(w[j]^2 * I - A) * dzdt[j]
        end
        S = -8 * KK * Complex(m * M)^(1/4) * imag.(S) .* A / (k * π * N)
    end
    return S
end

using LinearAlgebra
using Plots
using ForwardDiff
using LaTeXStrings
using Symbolics
using SparseArrays
using SparseDiffTools
using Elliptic
using Elliptic.Jacobi: sn, cn, dn
using Random, Distributions
using SymbolicUtils
include("envelope.jl")

n= 800

##Boundary Conditions
c_0 = Matrix{Float64}(I, n, n)
for i in 1:(n-1)
    c_0[i+1, i] = 1
    c_0[i, i+1] = 1
end

c_0[1,n]=1;
c_0[n,1]=1;

c1 = copy(c_0); 
c1[1,2] += 1.0; 
c1[2,1] += 1.0; 


## Define the energy function

function energy(c, u) 
    N = length(u) 
    E = sum( c[i,i+1] * (u[i+1]-u[i])^2 for i = 1:N-1 )+ 0.5* sum((u[i+1]-u[i])^3 for i = 1:N-1) + sum((u[i+1]-u[i])^4 for i = 1:N-1) + 
    c[N,1] * (u[1]-u[N])^2 + 0.5* (u[1]-u[N])^3 + (u[1]-u[N])^4
    return E
 end


hessian_hom(c) = ForwardDiff.hessian(u -> energy(c, u), zeros(n))
H0 = hessian_hom(c_0);


λ0, V0  = eigen(H0)
λ0[ abs.(λ0) .< 1e-10 ]  .= 1 
H0_= V0 * Diagonal(λ0) * V0'
lh0=log(H0_)
s0 = 0.5 * log(H0_)[400,400]

F = V0 * Diagonal(sqrt.(λ0)) * V0'; #F:=H^(1/2)
rtH0 = V0 * Diagonal(1 ./ sqrt.(λ0)) * V0';  #rtH0:=H^(-1/2)


##### Normalize
Random.seed!(123) # Setting the seed
d = Normal(0.0, 0.1)
x = rand(d, 800)
uu= F * x


hessian(c) = ForwardDiff.hessian(u -> energy(c, u), uu)
H1 = hessian(c1);
λ1, V1  = eigen(H1)
λ1[ abs.(λ1) .< 1e-10 ]  .= 1 
H1_ = V1 * Diagonal(λ1) * V1';


hcalls=0
function hess!(out, u)
    global hcalls += 1
    H = ForwardDiff.hessian(v -> energy(c1, v), u)
    out .= vec(H)
    nothing
end


## It makes the calculations much faster to use Symbolics to detect 
## the sparsity pattern first and then compute the Jacobian using forwarddiff_color

@variables u[1:800]
u_vec = [u[i] for i in 1:800]
energy_expr = energy(c1, u_vec)
H = Symbolics.hessian(energy_expr, u_vec)

sparsity_pattern=Symbolics.jacobian_sparsity(vec(H), u_vec) ## Sparsity Pattern
jac = Float64.(sparse(sparsity_pattern))
colors = matrix_colors(jac)
Jac_fin=forwarddiff_color_jacobian!(jac, hess!, uu, colorvec = colors)

F_l = F[:, 1]

### Perform fast Contour integration

function Contour_integral(dhdu)
    e =  eigen(F' * H1_ * F).values
    m = e[argmin(abs.(e))]
    M = e[argmax(abs.(e))]
    k = (Complex(M/m)^(1/4) - 1) / (Complex(M/m)^(1/4) + 1)
    L = -log(k) / π
    KK, Kp = ellipkkp(L)
    t = 0.5im * Kp .- KK .+ (0.5:15) .* (2 * KK / N)
    u, c, d = ellipjc(t, L)
    w = Complex(m * M)^(1/4) .* ((1/k .+ u) ./ (1/k .- u))
    dzdt = c .* d ./ (1/k .- u).^2
    S = 0
    for j in 1:15
        L, U= lu(w[j]^2 .* H0_- H1_)
        P= U \ (L \ F_l)
        S = S + (f(w[j]^2) * w[j]) * (P'* dhdu * P) * dzdt[j]
    end
    S = -8 * KK * (m * M)^(1/4) * imag(S) / (k * pi * N);
    return S
end

T_ = reshape(Jac_fin, 800,800,800)
ds_=zeros(n)

for k in 1:800 
    ds_[k] = Contour_integral(T_[:,:,k])
    print(k)
end

x = [ min(i, 1+n-i) for i = 1:n ];


#### Use envelope to get max points in respective intervals

ξ = 10 .^ (range(log10(6), stop=log10(300/1.2), length= 10))
y0 = abs.(ds_)
xe, ye0 = envelope(x, y0, ξ)
x0 = x


plt = Plots.scatter(grid=false, lw=0, m=:o,
                 dpi=400, size=(400,400),
                 legend=:bottomleft,
                 legendfontsize=9, 
                 xtickfont=font(9, "Times New Roman"),  
                 ytickfont=font(9, "Times New Roman"),  
                 xguidefont=font(10, "Times New Roman"), 
                 yguidefont=font(10, "Times New Roman"),)

scatter!(plt, x0, y0, m=:o, color="darkseagreen1", markersize=4, 
         markerstrokewidth=0,  
         yscale=:log10, xscale=:log10, xlabel=L"r_{nl} \; [\AA]",  
         ylabel=L"\frac{\partial S_{\ell}}{\partial u_n}", label=nothing, dpi=400, alpha=0.5)

scatter!(plt, xe[3:end], ye0[3:end], m=:o, color=3, linewidth=2, markersize=5, 
         markerstrokecolor=3, 
         markerstrokewidth=0.5, 
         label=L"\mathrm{\frac{\partial S_{\ell}}{\partial u_n}}")

tt = [15,240]
plot!(plt, tt, 3*tt.^(-2), lw=2,ls=:dash, label= L"r_{nl}^{-1}", color= 4)

savefig(plt, "1D_envelope.pdf")

##### Convergence of truncated site entropies wrt rcut
lh0=log(H1_)
s1 = 0.5 * log(H1_)[400,400]

S_S=[]
for rc in 1:12
    u_tr = zeros(800)
    u_tr[400-rc:400+rc] = uu[400-rc:400+rc]
    hessian(c) = ForwardDiff.hessian(u -> energy(c, u), u_tr)
    H1 = hessian(c1);
    λ1, V1  = eigen(H1)
    λ1[ abs.(λ1) .< 1e-10 ]  .= 1 
    H1_ = V1 * Diagonal(λ1) * V1';
    s0 = 0.5 * log(H1_)[400,400]
    push!(S_S, s0)
end


diff3=  abs.(S_S .- s1)
rc = 1:12


plt_S= Plots.plot(grid=false, lw=0, m=:o, ms=2,
xlabel=L"\mathrm{r_{cut}}",
ylabel=L"\mathrm{ACE\ predicted\ site\ entropies\ [k_B]}",
label=L"RMSE = 0.4",
dpi=400, size=(400,400),
legend=:topright,
xscale=:log10, yscale=:log10)

scatter!(plt_S, rc, diff, ms=5, label=L"δ = 0", color=1)
scatter!(plt_S, rc, diff1, ms=5, label=L"δ = 0.1", color=2)
scatter!(plt_S, rc, diff3, ms=5, label=L"δ = 0.5", color=3)

plot!(plt_S, rc[4:end],  0.12.*rc[4:end].^(-1.2), line=(:dash, 2), color=4)
savefig(plt_S, "site_entropy.pdf")

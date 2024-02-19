using JuLIP, ASE, Plots, ForwardDiff, LinearAlgebra, LaTeXStrings
using ACEpotentials
using Random, Arpack, Distributions, StaticArrays


sw = StillingerWeber()
bulk_Si = bulk(:Si, cubic=true, pbc=true) * (2,2,1)
Hessian_0 = hessian(sw, bulk_Si)

## shift .< 1e-10 eigenvalues to 1
λ0, ϕ0 = eigen(Array(Symmetric(Hessian_0)))
λ0[ λ0 .< 1e-10 ] .= 1 
H0_ = ϕ0 * Diagonal(λ0) * ϕ0'
lh0=log(H0_)
s0 = [0.5 * tr((lh0[(3*l)-2:3l , (3*l)-2:3l])) for l in 1:length(bulk_Si)];


function site_entropy(pot,data)
    Hessian_3d = hessian(pot, data)
    λ, ϕ = eigen(Array(Symmetric(Hessian_3d)))
    λ[ λ .< 1e-10 ] .= 1 
    H_ = ϕ * Diagonal(λ) * ϕ'
    lh = log(H_)
    s= [0.5 * tr((lh[(3*l)-2:3l , (3*l)-2:3l])) for l in 1:length(data)]
    Δs= s .- s0
    return Δs
end

F = ϕ0 * Diagonal(sqrt.(λ0)) * ϕ0';
rtH0 = ϕ0 * Diagonal(1 ./ sqrt.(λ0)) * ϕ0'; 

function fd_entropy(at)
    out=[]
    at2= deepcopy(at)
    x= position_dofs(at)
    for i in 1:length(x)
        x_= deepcopy(x)
        _x= deepcopy(x)
        x_[i] += 1e-5
        _x[i] -=  1e-5
        at_=set_position_dofs!(at, x_) 
        _at=set_position_dofs!(at2, _x)
        s_=site_entropy(sw,at_)
        _s=site_entropy(sw,_at)
        fd= (-s_ .+ _s)/(2*1e-5)
        push!(out,fd)
    end
    frc = Matrix{Float64}(undef,length(at),length(x))
    for j in 1:length(x)
        for k in 1:length(at)
            frc[k, j]= out[j][k]
        end
    end
    return frc
end



function create_data(N)

    bulk_dat =  Vector{Any}(undef, N)
    vac_dat = Vector{Any}(undef, N);
    
    ## Bulk_rattle
    
    for i in 1:N
        at = bulk(:Si, cubic=true, pbc=true) * (2,2,1);
        rattle!(at,0.06 * rand())
        bulk_dat[i]= at
    end
    
    
    ##Vacancy
    
    at = bulk(:Si, cubic=true, pbc=true) * (2,2,1);
    at_vac= deleteat!(at ,1)
    vac= get_positions(at_vac);
    set_calculator!(at_vac, sw);
    minimise!(at_vac);
    vac_min=get_positions(at_vac);
    u= vac_min - vac
    v = bulk_Si[1]
    u_proj = vcat([v], u)
    x= get_positions(bulk_Si)
    x += u_proj
    for j in 1:N
        at1= deepcopy(bulk_Si)
        set_positions!(at1, x);
        rattle!(at1,0.1 * rand())
        vac_dat[j]= at1
    end
    
    
    
    data= vcat(bulk_dat, vac_dat);
    
    S= Vector{Any}(undef, 200)
    Forces= Vector{Any}(undef, 200)
    
    for i in 1:200
        at = data[i]
        S[i] = site_entropy(sw, at)
        Forces[i] = fd_entropy(at)
        print(i)
    end
    
    
    
    
    
    S_bulk = S[1:N];
    S_vac = S[N+1:end];
    F_bulk = Forces[1:N];
    F_vac = Forces[N+1:end];
    return bulk_dat, vac_dat, S_bulk, S_vac, F_bulk, F_vac

end





function fit_site_entropy(bo, deg, rcut, data_tr, S_tr, F_tr, n_train, data_ts, S_ts, F_test, n_test) 
    
    n_at = length(data_tr[1])
    model = acemodel(elements = [:Si],
                    order = bo,
                    totaldegree = deg,
                    rcut = rcut)

    basis = model.basis

    descriptors = []
    for atoms in data_tr
        struct_descriptor = site_descriptors(basis, atoms)
        push!(descriptors, struct_descriptor)
    end

    ∂descriptors = zeros(3*(n_at)^2*n_train,length(basis))

    q=1
    for (idx,atoms) in enumerate(data_tr)
        for k in 1:n_at
            struct_descriptor = site_energy_d(basis, atoms,k)
            for i in 1:length(basis)
                ∂descriptors[q:q+3*n_at-1,i]=reduce(vcat,struct_descriptor[i])
            end
            q +=3*n_at
        end
    end


    A = Matrix{Float64}(undef, n_train * length(data_tr[1]), length(basis))

    for i in 1:n_train
        for j in 1:n_at
            A[n_at*i-n_at+j,:]= descriptors[i][j]
        end
    end

    y_matrix = transpose(vcat(S_tr...))
    F_train_ = transpose(vcat(F_tr...))


    D_mat = zeros(n_at * n_train + 3*n_at^2*n_train, length(basis))
    Y_vec = zeros(n_at * n_train + 3*n_at^2*n_train)
    W= ones(n_train*n_at + n_train*3*n_at^2)

    for i in 1:n_train
        A_rows = (n_at*(i-1)+1):(n_at*i)   
        ∂_rows = (3*n_at^2*(i-1)+1):(3*n_at^2*i) 
        D_mat[((3*n_at+1)*n_at*(i-1)+1):((3*n_at+1)*n_at*(i-1)+n_at), :] = A[A_rows, :]
        D_mat[((3*n_at+1)*n_at*(i-1)+(n_at+1)):((3*n_at+1)*n_at*i), :] = ∂descriptors[∂_rows, :]
        Y_vec[((3*n_at+1)*n_at*(i-1)+1):((3*n_at+1)*n_at*(i-1)+n_at)] = y_matrix[A_rows]
        Y_vec[((3*n_at+1)*n_at*(i-1)+(n_at+1)):((3*n_at+1)*n_at*i), :] = F_train_[∂_rows]
        W[(3*n_at+1)*n_at*(i-1)+1:(3*n_at+1)*n_at*(i-1)+n_at] .= 100/sqrt((n_at ÷ 2))
    end


    solver = ACEfit.RRQR(; rtol = 1e-8, P = smoothness_prior(basis; p = 4))
    #solver = ACEfit.BLR()
    results = ACEfit.solve(solver, W .* D_mat, W .* Y_vec)

    C=results["C"]

    test_descriptors = []
    for atoms in data_ts
        struct_descriptor = site_descriptors(basis, atoms)
        push!(test_descriptors, struct_descriptor)
    end

    ∂descriptors_ts = zeros(3*(n_at)^2*n_test,length(basis))


    q=1
    for (idx,atoms) in enumerate(data_ts)
        for k in 1:n_at
            struct_descriptor = site_energy_d(basis, atoms,k)
            for i in 1:length(basis)
                ∂descriptors_ts[q:q+3*n_at-1,i]=reduce(vcat,struct_descriptor[i])
            end
            q +=3*n_at
        end
    end

    A_t = Matrix{Float64}(undef, n_test * length(data_ts[1]), length(basis))

    for i in 1:n_test
        for j in 1:n_at
            A_t[n_at*i-n_at+j,:]= test_descriptors[i][j]
        end
    end

    y_matrix_t = vec(transpose(vcat(S_ts...)))
    F_test_ = -1 .* vec(transpose(vcat(F_test...)))

    y_pred_s= A_t * C
    rmse_S = sqrt(mean((y_matrix_t - y_pred_s).^2))

    y_pred_f= ∂descriptors_ts *C
    rmse_F = sqrt(mean((F_test_ - y_pred_f).^2))

    return y_matrix_t,y_pred_s, F_test_,y_pred_f, rmse_S, rmse_F 

end

################# Hyperparameter tuning

N = 100


bos = [3, 4]
degs = [8, 10, 12, 14, 16]

cartesian_product = collect(product(bos, degs))
cartesian_product_array = [ [x[1], x[2]] for x in cartesian_product ]


i=1
j=1
val_error=zeros(2,5)
for bo in [3,4]
    for deg in [8, 10, 12, 14, 16]
        bulk_dat, vac_dat, S_bulk, S_vac, F_bulk, F_vac = create_data(N)
        training = vac_dat[1:70]
        validation = vac_dat[71:end]
        val_error[i,j] = fit_site_entropy(bo, deg, 5.5, training, S_vac[1:70], F_vac[1:70], 70, validation,
                                             S_vac[71:end], F_vac[71:end], 30)[5]
        i+=1
    end
    j+=1
end

min_value, linear_index = findmin(matrix)
(row, col) = Tuple(CartesianIndices(matrix)[linear_index])

BO, DEG = cartesian_product_array[row, col]


#######################
bulk_dat, vac_dat, S_bulk, S_vac, F_bulk, F_vac = create_data(N)
data_test= vcat(bulk_dat[1:50], vac_dat[51:end])
S_test= vcat(S_bulk[1:50], S_vac[51:end])
F_test= vcat(F_bulk[1:50], F_vac[51:end])

y_matrix_t,y_pred_s, F_test_,y_pred_f, rmse_S, rmse_F  = fit_site_entropy(BO, DEG, 5.506786, vac_dat[1:50], 
                            S_vac[1:50], F_vac[1:50], 50, data_test, S_test, F_test, 100)


plt_S= Plots.plot(grid=false, lw=0, m=:o, ms=2,
xlabel=L"\mathrm{Reference\ site\ entropies\ [k_B]}",
ylabel=L"\mathrm{ACE\ predicted\ site\ entropies\ [k_B]}",
                        label=L"RMSE = 0.4",
                        dpi=400, size=(400,400),
                        legend=:bottomright,
                        xtickfont=font(9, "Times New Roman"),  
                        ytickfont=font(9, "Times New Roman"),  
                        xguidefont=font(10, "Times New Roman"), 
                        yguidefont=font(10, "Times New Roman"),
                        legendfont=font(9, "Times New Roman"),
                        xlims=(-0.09, 0.07), ylims=(-0.09, 0.07))

start_point, end_point = -0.085, maximum(y_pred_s)+0.01
x_values = [start_point, end_point]
y_values = x_values
scatter!(plt_S, y_pred_s[1601:end], y_matrix_t[1601:end], ms=5, label=L"\mathrm{Vacancy}",color=3)
scatter!(plt_S, y_pred_s[1:1600], y_matrix_t[1:1600],  ms=5, label=L"\mathrm{Bulk}",color=1)
plot!(plt_S, x_values, y_values,  line=(:dash, 1.5), color=2, label=nothing)
savefig(plt_S, "site_entropy.pdf")


plt_f= Plots.plot(grid=false, lw=0, m=:o, ms=2,
                     xlabel=L"\mathrm{Reference\ site\ entropy\ derivatives\ [k_B/ \AA]}",
                     ylabel=L"\mathrm{ACE\ site\ entropy\ derivatives\ [k_B/ \AA]}",
                     label=L"RMSE = 0.4",
                     dpi=400, size=(400,400),
                     legend=:bottomright,
                     xtickfont=font(9, "Times New Roman"),  
                     ytickfont=font(9, "Times New Roman"),  
                     xguidefont=font(10, "Times New Roman"), 
                     yguidefont=font(10, "Times New Roman"),
                     legendfont=font(9, "Times New Roman"),
                     xlims=(-0.48, 0.43), ylims=(-0.48, 0.43))

start_point, end_point = -0.43, 0.41
x_values = [start_point, end_point]
y_values = x_values

scatter!(plt_f, y_pred_f[153601:end], F_test_[153601:end], ms=5, label=L"\mathrm{Vacancy}", color = 3)
scatter!(plt_f, y_pred_f[1:153600], F_test_[1:153600], ms=5, label=L"\mathrm{Bulk}",color=1)
plot!(plt_f, x_values, y_values, line=(:dash, 1.5), color=2, label=nothing)
savefig(plt_f, "site_forces.pdf")



### RMSE
rmse_S_bulk = print("S_bulk RMSE: ",sqrt(mean((y_matrix_t[1:1600] - y_pred_s[1:1600]).^2)))
rmse_F_bulk = print("F_bulk RMSE: ", sqrt(mean((y_pred_f[1:153600]- F_test_[1:153600]).^2)))
rmse_S_vac = print("S_vac RMSE: ",sqrt(mean((y_matrix_t[1601:end] - y_pred_s[1601:end]).^2)))
rmse_F_vac = print("F_vac RMSE: ",sqrt(mean((y_pred_f[153601:end]- F_test_[153601:end]).^2)))

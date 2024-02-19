using JuLIP, ASE, Plots, ForwardDiff, LinearAlgebra
using ACEpotentials, LaTeXStrings
using Random, Arpack, Distributions, StaticArrays
using JuLIP, ASE , Plots, LinearAlgebra, ForwardDiff
using ACEpotentials
using IterTools: product


sw = StillingerWeber()
bulk_Si = bulk(:Si, cubic=true, pbc=true) * (2,2,1);
Hessian_0 = hessian(sw, bulk_Si)

## shift .< 1e-10 eigenvalues to 1
λ0, ϕ0 = eigen(Array(Symmetric(Hessian_0)))
λ0[ λ0 .< 1e-10 ] .= 1 
H0_ = ϕ0 * Diagonal(λ0) * ϕ0'
s0= logabsdet(H0_)[1]

function entropy(pot,data)
    Hessian_3d = hessian(pot, data)
    λ, ϕ = eigen(Array(Symmetric(Hessian_3d)))
    λ[ λ .< 1e-10 ] .= 1 
    H_ = ϕ * Diagonal(λ) * ϕ'
    s = 1/2*(s0- (logabsdet(H_)[1]))
    return s
end

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
        s_=entropy(sw,at_)
        _s=entropy(sw,_at)
        fd= (-s_ + _s)/(2*1e-5)
        push!(out,fd)
    end
    return (reshape(out, (3,length(at))))'
end

F = ϕ0 * Diagonal(sqrt.(λ0)) * ϕ0';
rtH0 = ϕ0 * Diagonal(1 ./ sqrt.(λ0)) * ϕ0'; 


function create_data(N)

    bulk_dat =  Vector{Any}(undef, N)
    vac_dat = Vector{Any}(undef, N);

    ## Bulk data + rattled

    for i in 1:N
        at = bulk(:Si, cubic=true, pbc=true) * (2,2,1);
        rattle!(at,0.08 * rand())
        bulk_dat[i]= at
    end

    ## Vacancy data + rattled

    at = bulk(:Si, cubic=true, pbc=true) * (2,2,1);
    at_vac= deleteat!(at ,1)
    vac= get_positions(at_vac);
    set_calculator!(at_vac, sw);
    minimise!(at_vac);
    vac_min=get_positions(at_vac);
    u= vac_min - vac
    v = zeros(3)
    u_proj = vcat([v], u)
    x= get_positions(bulk_Si)
    x += u_proj
    for j in 1:N
        at1= deepcopy(bulk_Si)
        set_positions!(at1, x);
        rattle!(at1,0.1 * rand())
        vac_dat[j]= at1
    end

    #### Evaluating S and Force data

    bulk_vac= vcat(bulk_dat, vac_dat);

    S= Vector{Real}(undef, 200)
    Forces=[]

    for i in 1:200
        at = bulk_vac[i]
        S[i] = entropy(sw, at)
        push!(Forces, fd_entropy(at))
        print(i)
    end

    S_bulk = S[1:N];
    S_vac = S[N+1:end];
    F_bulk = Forces[1:N];
    F_vac = Forces[N+1:end];

    return bulk_dat, vac_dat, S_bulk, S_vac, F_bulk, F_vac

end

###Fitting
function fit_entropy(species, bo, deg, rcut, weights,
    data, S_tr, F_tr, n_train)

    model = acemodel(elements = [species],
    order = bo,
    totaldegree = deg,
    rcut = rcut)

    basis = model.basis


    datakeys = (energy_key = "energy", force_key = "force", virial_key = "nothing")

    new_data=[]

    for i in 1:n_train
        set_data!(data[i], datakeys.energy_key, S_tr[i])
        set_data!(data[i], datakeys.force_key, vecs(Float64.(F_tr[i]'))) 
        push!(new_data, data[i])
    end

    train = [ACEpotentials.AtomsData(t; weights=weights, datakeys...) for t in new_data] 

    A, Y, W = ACEfit.assemble(train, basis);
    solver = ACEfit.RRQR(; rtol = 1e-8, P = smoothness_prior(basis; p = 4));
    # solver = ACEfit.BLR()

    results = ACEfit.solve(solver, W .* A, W .* Y)
    pot = JuLIP.MLIPs.SumIP(JuLIP.MLIPs.combine(basis, results["C"]));

    C=results["C"]
    y_pred = A * C

    return pot

end


### Hyperparameter tuning
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
        weights = Dict("default" => Dict("E" => 100.0, "F" => 1.0 , "V" => 0.0 ));
        ace_model= fit_entropy(:Si, bo, deg, 5.5, weights, training, S_vac[1:0.7*N], F_vac[1:0.7*N], 0.7*N);
        datakeys = (energy_key = "energy", force_key = "force", virial_key = "nothing")
        val_test=[]
        for i in 1:30
            set_data!(validation[i], datakeys.energy_key, S_vac[70 + i])
            set_data!(validation[i], datakeys.force_key,  vecs(Float64.(F_vac[70 + i])'))
            push!(val_test, validation[i])
        end
        vac_val = [ACEpotentials.AtomsData(t; weights=weights, datakeys...) for t in val_test] 
        e_val = []
        for d in vac_val
            push!(e_val,energy(ace_model, d.atoms))
        end

        val_error[i,j] = sqrt(mean((e_val - S_bulk[71:end]).^2))
        i+=1
    end
    j+=1
end

min_value, linear_index = findmin(matrix)
(row, col) = Tuple(CartesianIndices(matrix)[linear_index])

BO, DEG = cartesian_product_array[row, col]

### Training

bulk_dat, vac_dat, S_bulk, S_vac, F_bulk, F_vac = create_data(N)

weights = Dict(
"default" => Dict("E" => 100.0, "F" => 1.0 , "V" => 0.0 ));

ace_model= fit_entropy(:Si, BO, DEG, 5.506786, weights, vac_dat[1:50], S_vac[1:50], F_vac[1:50], 50);


## Testing
datakeys = (energy_key = "energy", force_key = "force", virial_key = "nothing")

test_vac=[]
for i in 1:50
    set_data!(vac_dat[50+i], datakeys.energy_key, S_vac[50+i])
    set_data!(vac_dat[50+i], datakeys.force_key,  vecs(Float64.(F_vac[50+i])'))
    push!(test_vac, vac_dat[50+i])
end


test_bulk=[]
for i in 1:50
    set_data!(bulk_dat[i], datakeys.energy_key, S_bulk[i])
    set_data!(bulk_dat[i], datakeys.force_key,  vecs(Float64.(F_bulk[i])'))
    push!(test_bulk, bulk_dat[i])
end

bulk_test = [ACEpotentials.AtomsData(t; weights=weights, datakeys...) for t in test_bulk] 
vac_test = [ACEpotentials.AtomsData(t; weights=weights, datakeys...) for t in test_vac] 


## Evaluation
e_vac, f_vac = [],[]
for d in vac_test
    push!(e_vac,energy(ace_model, d.atoms))
    push!(f_vac,forces(ace_model, d.atoms))
end

e_bulk,f_bulk = [],[]
for d in bulk_test
    push!(e_bulk,energy(ace_model, d.atoms))
    push!(f_bulk,forces(ace_model, d.atoms))
end

### Some reshaping to plot easier

bulk_F_exact = vcat(vcat(F_bulk[1:50]...)...)
bulk_F_pred = vcat([[x[i] for x in vcat(f_bulk...)] for i in 1:3]...);

vac_F_exact = vcat(vcat(F_vac[51:end]...)...)
vac_F_pred = vcat([[x[i] for x in vcat(f_vac...)] for i in 1:3]...);


plt_S= Plots.plot(grid=false, lw=0, m=:o, ms=2,
                     xlabel=L"\mathrm{Reference\ entropy\ derivatives\ [k_B/ \AA]}",
                     ylabel=L"\mathrm{ACE\ entropy\ derivatives\ [k_B/ \AA]}",
                     label=L"RMSE = 0.4",
                     dpi=400, size=(400,400),
                     legend=:bottomright,
                     xtickfont=font(9, "Times New Roman"),  
                     ytickfont=font(9, "Times New Roman"),  
                     xguidefont=font(10, "Times New Roman"), 
                     yguidefont=font(10, "Times New Roman"),
                     legendfont=font(9, "Times New Roman"))

start_point, end_point = minimum(vac_F_exact)-0.1, maximum(vac_F_exact)+0.1
x_values = [start_point, end_point]
y_values = x_values
scatter!(plt_S, vac_F_exact, vac_F_pred, ms=5, label=L"\mathrm{Vacancy}",color=3)
scatter!(plt_S, bulk_F_exact, bulk_F_pred, ms=5, label=L"\mathrm{Bulk}",color=1)
plot!(plt_S, x_values, y_values, line=(:dash, 1.5), color=2, label=nothing)
savefig(plt_S, "Test_entropy_derivatives.pdf")

# data_S_vac = DataFrame(X=vac_F_exact, Y=vac_F_pred)
# data_S_bulk = DataFrame(X=bulk_F_exact, Y=bulk_F_pred)
# CSV.write("Total_S_vac.csv", data_S_vac)
# CSV.write("Total_S_bulk.csv", data_S_bulk)


plt_F = plot(grid=false, lw=0, m=:o, ms=5,
                 xlabel=L"\mathrm{Reference\ entropies\ [k_B]}",
                 ylabel=L"\mathrm{ACE\ entropies\ [k_B]}",
                 label=L"S_l",
                 dpi=400, size=(400,400),
                 legend=:bottomright,
                 xtickfont=font(9, "Times New Roman"),  
                 ytickfont=font(9, "Times New Roman"),  
                 xguidefont=font(10, "Times New Roman"), 
                 yguidefont=font(10, "Times New Roman"), 
                 legendfont=font(9, "Times New Roman"))

start_point, end_point = minimum(e_bulk)-0.01, maximum(e_vac)+0.02
x_values = [start_point, end_point]
y_values = x_values
scatter!(plt_F, e_vac, S_vac[51:end], ms=5, label=L"\mathrm{Vacancy}", color = 3)
scatter!(plt_F, e_bulk, S_bulk[1:50], ms=5, label=L"\mathrm{Bulk}",color=1)
plot!(plt_F, x_values, y_values, line=(:dash, 1.5), color=2, label=nothing)
savefig(plt_F, "Test_entropy.pdf")

# data_dS_vac = DataFrame(X=e_vac, Y=S_vac[21:end])
# data_dS_bulk = DataFrame(X=e_bulk, Y=S_bulk[21:end])
# CSV.write("Total_dS_vac.csv", data_dS_vac)
# CSV.write("Total_dS_bulk.csv", data_dS_bulk)



### RMSE
rmse_S_bulk = print("S_bulk RMSE: ",sqrt(mean((e_bulk - S_bulk[1:50]).^2)))
rmse_F_bulk = print("F_bulk RMSE: ", sqrt(mean((bulk_F_exact- bulk_F_pred).^2)))
rmse_S_vac = print("S_vac RMSE: ",sqrt(mean((e_vac - S_vac[51:end]).^2)))
rmse_F_vac = print("F_vac RMSE: ",sqrt(mean((vac_F_exact- vac_F_pred).^2)))
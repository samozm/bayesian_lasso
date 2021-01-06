using Random, Distributions, GLM, DataFrames, LinearAlgebra
using CSV
using Plots


"""
    update_σ!(σ², n,p, ỹ, β, D, X)

Update the value of σ² by sampling from the InverseGaussian distribution. 
Maintains previous values of σ² by adding the new value to the end of the current σ² vector

# Arguments 
- `σ²`: a vector containing all previous values of σ². The current value of σ² is last.
- `n`: the total number of observations used to fit the model
- `p`: the total number of parameters used in the model
- `ỹ`: the centered y value 
- `β`: a dataframe with all previous values of β. The current value of β (to be used) is 
    the last row.
- `D`: Diagonal matrix of τ² values
- `X`: matrix containing actual data (excluding response variable)

# Returns     
- nothing
"""
function update_σ!(σ², n, p, ỹ, β, D, X)
    a = (n-1+p)/2

    # b = [ (ỹ - Xβᵀ)ᵀ(ỹ - Xβᵀ) + βD⁻¹βᵀ ] / 2
    ỹ_X_β = ỹ - X * transpose(β)
    b = ( (transpose(ỹ_X_β) * (ỹ_X_β) + (β * inv(D) * transpose(β))) / 2 )[1]

    # sample from InverseGamma(a,b)
    new_σ² = rand(InverseGamma(a,b))
    append!(σ²,new_σ²)
end

"""
    update_β!(β, X, D, σ², ỹ)

Update the value of β by sampling from the MultivariateNormal distribution. 
Maintains previous values of β by adding the new values as a new row to the
end of the current β dataframe.

# Arguments 
- `β`: a dataframe with all previous values of β. The new value of β is appended as
    the last row.
- `X`: matrix containing actual data (excluding response variable)
- `D`: Diagonal matrix of τ² values
- `σ²`: a vector containing all previous values of σ². The current value of σ² is last.
- `ỹ`: the centered y value 

# Returns     
- nothing
"""
function update_β!(β, X, D, σ², ỹ)
    D⁻¹ = inv(D)

    # A = (XᵀX + D⁻¹) but we only care about A⁻¹ so we'll take the inverse right away
    A⁻¹ = inv(transpose(X)*X + D⁻¹)

    # for sampling β: μᵦ = A⁻¹ Xᵀ ỹ
    μ = A⁻¹ * transpose(X) * ỹ

    # for sampling β: σᵦ² = σ² A⁻¹
    var = round.(last(σ²) .* A⁻¹, digits=0)

    # sample from MVN(μᵦ, σᵦ²)
    new_β = rand(MultivariateNormal(μ,var))
    push!(β,new_β)
end

"""
    update_τ!(τ, λ, β)

Update the value of τ by sampling from the InverseGaussian distribution. 
Maintains previous values of τ by adding the new values as a new row to the
end of the current τ dataframe.

# Arguments 
- `τ`: a dataframe with all previous values of τ. The new value of τ is appended as
    the last row.
- `λ`: current value of λ (penalty term)
- `β`: a dataframe with all previous values of β. The current value of β (to be used) is 
    the last row.

# Returns     
- nothing
"""
function update_τ!(τ, λ, β)
    p = size(β,2)
    new_τ = zeros(p)

    #TODO: is there a way to do this without a for loop?
    for i in 1:p
        # μₜₐᵤ = √λ̅²̅/̅β̅ᵢ²
        μ = sqrt((λ^2) / (β[1,i]^2))
        # σₜₐᵤ = λ²
        var = λ^2

        new_τ[i] = 1 / rand(InverseGaussian(μ,var))
    end
    push!(τ, new_τ)
end

"""
    update_λ!(λ, exp_τ, p)

Update λ using the expected value of τ².

# Arguments
- `λ`: a vector containing all previous values of λ. The new value of λ is appended to the end.
- `exp_τ`: expexted value of τ, based on most recent gibbs samples
- `p`: the total number of parameters used in the model
"""
function update_λ!(λ, exp_τ, p)
    new_λ = sqrt( (2*p) / sum(exp_τ) )
    append!(λ, new_λ)
end

"""
    gibbs_sample!(β, σ², τ², y, X, n, p, λ)

Take one gibbs sample. 

# Arguments 
- `β`: a dataframe with all previous values of β. The current value of β (to be used) is 
    the last row.
- `σ²`: a vector containing all previous values of σ². The current value of σ² is last.
- `τ²`: a dataframe with all previous values of τ. The current value of τ is the last row.
- `y` : the response variable
- `X` : matrix containing actual data (excluding response variable)
- `n` : the total number of observations used to fit the model
- `p` : the total number of parameters used in the model 
- `λ` : a vector containing all previous values of λ. The current value of λ is the last.

# Returns     
- nothing
"""
function gibbs_sample!(β, σ², τ², ỹ, X, n, p, λ)
    D = Diagonal(vec(convert(Array,last(τ²,1))))
    update_β!(β, X, D, last(σ²), ỹ)
    update_σ!(σ², n, p, ỹ, convert(Array,last(β,1)), D, X)
    update_τ!(τ², last(λ), convert(Array,last(β,1)))
end

"""
    create_df(df)
    
Take in a column dataframe taken as a sample from a distribution and make it usable by 
transposing it.

# Arguments
- `df`: column dataframe generated from a distribution

# Returns
- row dataframe (transpose of input)
"""
function create_df(df)
    df[!, :id] = 1:size(df, 1)
    tmp = stack(df)
    df = select!(unstack(tmp, :id, :value), Not(:variable))
    return(df)
end

"""
    plot_results(β, olm, colnm)

Plot a histogram for each parameter in β against the least squares value

# Arguments
- `β`    : a dataframe with values of β to plot (post burn-in samples)
- `olm`  : output of lm function, the result of performing ordinary least squares on the data
- `colnm`: array of column names taken from the data, used for plot titles

# Returns
- nothing
"""
function plot_results(β, olm, colnm)
    # use gr as plot backend
    gr()

    # Initialize empty array to put plots in 
    histo = Array{Plots.Plot{Plots.GRBackend},1}()
    for i in 1:size(β, 2)
        # plot histogram with line for OLS value
        p = histogram(β[!,i], label="BLasso", title=colnm[i], size=(1600,1200))
        vline!(p, [coef(olm)[i+1]], label="OLS")
        push!(histo,p)
    end
    plot(histo..., layout=size(histo,1))
end

function main()
    # let's use diabetes dataset from the original paper
    data = DataFrame!(CSV.File("diabetes.txt", delim=" "))

    # run ols to get initial value for lambda
    olm = lm(@formula(y~age+sex+bmi+map+tc+ldl+hdl+tch+ltg+glu), data)

    # Initialize y matrix and n from data
	n = size(data, 1)
    y = data[!,:y]
    ỹ = y .- mean(y)

    # Initialize X matrix from data
    X = convert(Matrix, select(data, Not(:y)))
    p = size(X, 2)

    # Initialize β, τ² - in monomvn these are initialized to 0
    β = create_df(DataFrame(B = zeros(p)))

    # do we need to use that distribution of σ², τ² from the full model? 
    τ² = create_df(DataFrame(B = ones(p)))

    # Initialize σ²
    # monomvn just uses var(y - mean(y))
    σ² = [var(ỹ)]

    # Initialize λ using OLS
    sum_exp = sum(abs.(coef(olm)))
    λ = [p*sqrt(last(σ²))/sum_exp]

    # 1000 burn-in samples
    for i in 1:1000
        gibbs_sample!(β, σ², τ², ỹ, X, n, p, λ)
    end

    # 50000 Gibbs samples
    for i in 1:100
        for j in 1:500
            gibbs_sample!(β, σ², τ², ỹ, X, n, p, λ)
        end
        exp_τ = sum(convert(Matrix,last(τ²,100)), dims = 1)./100 
        update_λ!(λ, sum(exp_τ), p)
    end

    plot_results(last(β,500), olm, names(select(data, Not(:y))))
    gui()

end

main()
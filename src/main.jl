"""
Transform x::Real to be in a [0.1] interval.
"""
@inline transform_unit_interval(x::Real) = 1.0 / (exp(-x) + 1.0)
@inline transform_from_unit_interval(x::Real) = logit(x)
@inline invlogit(x::Real) = 1.0 / (exp(-x) + 1.0)
@inline transform_unit_interval(x::Real, a::Real, b::Real) = (b - a) * invlogit(x) + a
@inline transform_from_unit_interval(x::Real, a::Real, b::Real) = logit((x - a) / (b - a))

const LOWER_ALPHA = 1e-04
const UPPER_ALPHA = 1 - 1e-04

const LOWER_BETA = 1e-04
const UPPER_BETA = 1 - 1e-04


abstract type AbstractExponentitalSmoothing end

struct FOES{VX, IDX, MX, R} <: AbstractExponentitalSmoothing
    parms::VX
    residuals::VX
    states::MX
    values::VX
    index::IDX
    sigma::R
end

struct Holt{VX, IDX, MX, R} <: AbstractExponentitalSmoothing
    parms::VX
    residuals::VX
    states::MX
    values::VX
    index::IDX
    sigma::R
end

struct HoltDumped{VX, IDX, MX, R} <: AbstractExponentitalSmoothing
    parms::VX
    residuals::VX
    states::MX
    values::VX
    index::IDX
    sigma::R
end

function FOES(y::T) where T<:AbstractVector
    ly = length(y)
    m = FOES(Array{Float64, 1}(undef, 2),
             Array{Float64, 1}(undef, ly),
             Array{Float64, 2}(undef, ly+1, 1),
             y,
             1:length(y),
             Ref{Float64}(NaN))
    startx!(m)
    m
end

function FOES(y::T) where T<:AbstractTimeSeries
    ly = length(y)
    m = FOES(Array{Float64, 1}(undef, 2),
             Array{Float64, 1}(undef, ly),
             Array{Float64, 2}(undef, ly+1, 1),
             float(values(y)),
             timestamp(y),
             Ref{Float64}(NaN))
    startx!(m)
    m
end

function Holt(y::T) where T<:AbstractVector
    ly = length(y)
    m = Holt(Array{Float64, 1}(undef, 4),
             Array{Float64, 1}(undef, ly),
             Array{Float64, 2}(undef, ly+1, 2),
             y,
             1:length(y),
             Ref{Float64}(NaN))
    startx!(m)
    m
end

function Holt(y::T) where T<:AbstractTimeSeries
    ly = length(y)
    m = Holt(Array{Float64, 1}(undef, 4),
             Array{Float64, 1}(undef, ly),
             Array{Float64, 2}(undef, ly+1, 2),
             float(values(y)),
             timestamp(y),
             Ref{Float64}(NaN))
    startx!(m)
    m
end


next(idx::StepRange{D, P}, h) where {D<:Date, P<:DatePeriod} = next(collect(idx), h)

function next(idx::Array{Date, 1}, h)
    ## Try to infer the frequency of the series
    mts = maximum(diff(idx))
    if mts >= Day(365)
        last(idx) + Year(1):Year(1):last(idx) + Year(h)
    elseif mts >= Day(1) & mts <= Day(4)
        last(idx) + Day(1):Day(1):last(idx) + Day(h)
    elseif mts >= Day(90)
        last(idx) + Month(3):Month(3):last(idx) + Month(3*h)
    end
end

next(idx::UnitRange{P}, h) where P<:Int64 = last(idx) + 1:last(idx) + h
next(idx::AbstractVector{P}, h) where P<:Int64 = last(idx) + 1:last(idx) + h


struct Forecast{MOD, A, M, V, L}
    method::MOD
    mean::A
    margin::M
    sigma::V
    level::L
end

function trendcoef(y)
    T = length(y)
    m = mean(y)
    tbar = (1 + T)/2
    s = 0.0
    @simd for j in eachindex(y)
        @inbounds s += y[j]*j
    end
    b1 = (s - T*tbar*m)/(T*(T^2-1)/12)
    (m - tbar*b1, b1)
end


#---
# FOES - First Order Exponential Smoothing
#---

objective(m::FOES) = begin
    (y, θ) -> begin
        ly = length(y)
        a  = transform_unit_interval(first(θ), 1e-04, 0.9999)
        s₀ = last(θ)
        s = a*y[1] + (1-a)*s₀
        e = abs2(s - y[2]) + abs2(y[1] - s₀)
        @inbounds for t in 2:ly-1
            s = a*y[t] + (1-a)*s
            e += abs2(s - y[t+1])
        end
        e
    end
end

function startx!(m::FOES)
    m.parms[1] = transform_from_unit_interval(.5, 1e-04, 0.9999)
    m.parms[2] = mean(m.values)
end

function recursion!(m::FOES)
    ly = length(m.values)
    a, s₀ = m.parms
    m.states[1] = s₀
    m.residuals[1] = m.values[1] - s₀
    for t in 1:ly-1
        m.states[t+1] = a*m.values[t] + (1-a)*m.states[t]
        m.residuals[t+1] = m.values[t+1] - m.states[t+1]
    end
    m.states[ly+1] = a*m.values[ly] + (1-a)*m.states[ly]
end

function var_prediction(m::FOES, h)
    σ² = m.sigma.x
    a  = m.parms[1]
    (σ²*(1 + a^2*(j-1)) for j in 1:h)
end

function mean_forecast(m::FOES, h)
    meanf = Array{Float64,1}(undef, h)
    fill!(meanf, last(m.states))
end


# ---
# HOLT - Second Order Exponential Smoothing
# ---

objective(m::Holt) = begin
    (y, θ) -> begin
        ## θ = [α, β, ℓ₀, s₀]
        ly = length(y)
        a  = transform_unit_interval(θ[1], 1e-04, 0.9999)
        b  = transform_unit_interval(θ[2], 1e-04, 0.9999)
        ℓ₋₁ = θ[3]
        s₋₁ = θ[4]
        e = abs2(y[1] - ℓ₋₁ - s₋₁)
        @inbounds for t in 1:ly-1
            ℓ = a*y[t] + (1-a)*(ℓ₋₁ + s₋₁)
            s = b*(ℓ - ℓ₋₁) + (1-b)*s₋₁
            ℓ₋₁ = ℓ
            s₋₁ = s
            e += abs2(y[t+1] - ℓ - s)
        end
        e
    end
end


function recursion!(m::Holt)
    ly = length(m.values)
    a, b, ℓ₀, s₀ = m.parms
    m.states[1, 1] = ℓ₀
    m.states[1, 2] = s₀
    m.residuals[1] = m.values[1] - ℓ₀ - s₀
    for t in 1:ly-1
        m.states[t+1, 1] = a*m.values[t] + (1-a)*(m.states[t, 1] + m.states[t, 2])
        m.states[t+1, 2] = b*(m.states[t+1, 1] - m.states[t, 1]) + (1-b)*m.states[t, 2]
        m.residuals[t+1] = m.values[t+1] - m.states[t+1, 1] - m.states[t+1, 2]
    end
    m.states[ly+1, 1] = a*m.values[ly] + (1-a)*(m.states[ly, 1] + m.states[ly, 2])
    m.states[ly+1, 2] = b*(m.states[ly+1, 1] - m.states[ly, 1]) + (1-b)*m.states[ly, 2]
end

function startx!(m::Holt)
    m.parms[1] = transform_from_unit_interval(.5, 1e-04, 0.9999)
    m.parms[2] = transform_from_unit_interval(.5, 1e-04, 0.9999)
    b0, b1 = trendcoef(m.values)
    m.parms[3] = b0
    m.parms[4] = b1
end

function var_prediction(m::Holt, h)
    σ² = m.sigma.x
    a, b, l₀, s₀ = m.parms
    (σ²*(1 + (j-1)*(a^2 + a*b*j + (1/6)*(b^2)*j*(2*j -1))) for j in 1:h)
end

function mean_forecast(m::Holt, h)
    meanx = Array{Float64,1}(undef, h)
    ℓ₋₁ = m.states[end,1]
    s₋₁ = m.states[end,2]
    a, b, l₀, s₀ = m.parms
    for j in 1:h
        meanx[j] = ℓ₋₁ + j*s₋₁
    end
    meanx
end

#=
HOLT - (Dumped) Second Order Exponential Smoothing
=#

holt_dumped_obj(y, θ) = begin
    ## θ = [α, β, ℓ₀, s₀, ϕ]
    a  = transform_unit_interval(θ[1])
    b  = transform_unit_interval(θ[2])
    ℓ₀ = θ[3]
    s₀ = θ[4]
    ℓ = a*y[1] + (1-a)*(ℓ₀ + s₀)
    s = b*(ℓ - ℓ₀) + (1-b)*s₀
    ℓ₋₁ = ℓ
    s₋₁ = s
    e = abs2(y[1] - ℓ₀ - ϕ*s₀) + abs2(y[2] - ℓ - ϕ*s)
    @inbounds for t in 2:ly-1
        ℓ = a*y[t] + (1-a)*(ℓ₋₁ + ϕ*s₋₁)
        s = b*(ℓ - ℓ₋₁) + (1-b)*ϕ*s₋₁
        ℓ₋₁ = ℓ
        s₋₁ = s
        e += abs2(y[t+1] - ℓ - s)
    end
    e
end

function prediction_intervals(m::T, h::Int, level::AbstractArray) where T<:AbstractExponentitalSmoothing
    v = var_prediction(m, h)
    meanf  = mean_forecast(m, h)
    margin = (norminvcdf(1-(1-j)/2).*sqrt.(v) for j in level)
    return meanf, v, margin, level
end

function Forecast(m::T; h = 12, level = [.95]) where T<:AbstractExponentitalSmoothing
    meanf, v, margin, level = prediction_intervals(m, h, level)
    Forecast(m, meanf, margin, v, level)
end

function foes(y::T) where T<:AbstractVector
    m = FOES(y)
    obj = objective(m)
    res = optimize(θ -> obj(m.values, θ), m.parms, BFGS(), autodiff = :forward)
    m.parms[1] = transform_unit_interval(Optim.minimizer(res)[1], 1e-04, 0.9999)
    m.parms[2] = Optim.minimizer(res)[2]
    recursion!(m)
    m.sigma.x = var(m.residuals)
    m
end

function foes(y::T) where T<:AbstractTimeSeries
    m = FOES(y)
    obj = objective(m)
    res = optimize(θ -> obj(m.values, θ), m.parms, BFGS(), autodiff = :forward)
    m.parms[1] = transform_unit_interval(Optim.minimizer(res)[1], 1e-04, 0.9999)
    m.parms[2] = Optim.minimizer(res)[2]
    recursion!(m)
    m.sigma.x = var(m.residuals)
    m
end

function holt(y::T) where T<:AbstractVector
    m = Holt(y)
    obj = objective(m)
    #res = optimize(θ -> obj(m.values, θ), m.parms, BFGS(), autodiff = :forward)
    res = optimize(θ -> obj(m.values, θ), m.parms, NelderMead())
    m.parms[1] = transform_unit_interval(Optim.minimizer(res)[1], 1e-04, 0.9999)
    m.parms[2] = transform_unit_interval(Optim.minimizer(res)[2], 1e-04, 0.9999)
    m.parms[3] = Optim.minimizer(res)[3]
    m.parms[4] = Optim.minimizer(res)[4]
    recursion!(m)
    m.sigma.x = var(m.residuals)
    m
end

function holt(y::T) where T<:AbstractTimeSeries
    m = Holt(y)
    obj = objective(m)
    res = optimize(θ -> obj(m.values, θ), m.parms, BFGS(), autodiff = :forward)
    m.parms[1] = transform_unit_interval(Optim.minimizer(res)[1], 1e-04, 0.9999)
    m.parms[2] = transform_unit_interval(Optim.minimizer(res)[2], 1e-04, 0.9999)
    m.parms[3] = Optim.minimizer(res)[3]
    m.parms[4] = Optim.minimizer(res)[4]
    recursion!(m)
    m.sigma.x = var(m.residuals)
    m
end




@recipe function fanchart(f::T; forecast_linecolor = :white, colorscheme = ColorSchemes.Blues_3, label = "") where T<:Forecast
    seriestype  --> :path
    insmpl = f.method.index
    outsmpl = next(f.method.index, length(f.mean))
    full_idx = [f.method.index; next(f.method.index, length(f.mean))]
    c1 = colorant"#005073"
    c2 = colorant"#71c7ec"
    col_margin = range(c1, stop=c2, length=length(f.margin))
    @series begin
        linecolor --> :red
        seriestype := :path
        insmpl, f.method.values
    end

    for (i, mar) in enumerate(Iterators.reverse(f.margin))
        @series begin
            linecolor := get(colorscheme, maximum(f.level))
            fillcolor := get(colorscheme, f.level[i])
            ribbon := (mar, mar)
            outsmpl, f.mean
        end
    end

    x = length(insmpl) + 1; y = 1
    h = Inf; w = length(full_idx) - x
    @series begin
        seriestype := :shape
        fillcolor --> :lightgray
        fillalpha := 0.4
        linewidth := 0.0
        x .+ [0, w, w, 0], y .+ [-Inf, -Inf, h, h]
    end

    ## Plot the anchor
    @series begin
        seriestype := :path
        linecolor --> :red
        linestyle := :dot
        [last(insmpl); first(outsmpl)], [last(f.method.values); first(f.mean)]
    end

    ## Plot 

end


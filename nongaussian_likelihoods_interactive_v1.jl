### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c9603eb0-0407-11ec-1751-3de72d9fd996
begin
	import Pkg
	
	# activate a temporary environment
	Pkg.activate(mktempdir())
	Pkg.add([
		Pkg.PackageSpec(name="Plots", version="1"),
		Pkg.PackageSpec(name="PlutoUI", version="0.7"),
		Pkg.PackageSpec(name="Distributions", version="0.25"),
		Pkg.PackageSpec(name="StatsBase", version="0.33"),
		Pkg.PackageSpec(name="StatsPlots", version="0.14"),
		Pkg.PackageSpec(name="AbstractGPs", version="0.5"),
		Pkg.PackageSpec(name="LogExpFunctions", version="0.3"),
		Pkg.PackageSpec(name="QuadGK", version="2"),
		Pkg.PackageSpec(name="ForwardDiff", version="0.10"),
		Pkg.PackageSpec(name="KLDivergences", version="0.2"),
		Pkg.PackageSpec(url="https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl/", rev="master"),  # waiting to be registered
    ])

	using LinearAlgebra
	using Random

	using Plots
	using PlutoUI
	using Distributions
	using StatsBase
	using StatsPlots
	using AbstractGPs
	using LogExpFunctions
	using QuadGK
	using ForwardDiff
	using KLDivergences
	using ApproximateGPs
end

# ╔═╡ 3a6a35fd-9acd-49db-9249-d8850fa1e275
md"""
# Gaussian processes for non-Gaussian likelihoods

Copyright (c) 2021 by ST John, [infinitecuriosity.org](http://infinitecuriosity.org)

The code in this notebook is licensed under the [MIT license](https://opensource.org/licenses/MIT).
"""

# ╔═╡ 52074a94-620e-417d-afc2-e9c05024069d
TableOfContents()

# ╔═╡ 0b7b7155-1bef-4070-81d1-b3a506d7fb7e
# plotly()
gr()

# ╔═╡ 4e22cdca-5180-4875-a43d-16efa21fc274
function slidesfig(fn)
	return plot!()
end

# ╔═╡ 507d6b85-92ab-458c-a6f1-46a5f9b178be
p_blank = plot(grid=false,foreground_color_subplot=:white);

# ╔═╡ 9ba94fc8-1951-4c21-9c3a-e655a70321e7
function covellipse2!(mean, cov; label="", kwargs...)
	covellipse!(mean, cov; label, n_std=1, kwargs...)
	covellipse!(mean, cov; label="", n_std=2, kwargs...)
end

# ╔═╡ 57cc2052-72c1-484a-88be-b3b216ff045b
function rand_gp_sample!(fx, f_at_x, xs, out)
	f_post = posterior(fx, [f_at_x...])
	rand!(f_post(xs, 1e-9), out)
end

# ╔═╡ c4825dde-ddf9-4a16-a9ea-c65a8eb50319
function get_gp_samples(xs, fx, f_at_x_samples)
	samples = length(f_at_x_samples)
	fs = zeros(samples, length(xs))
	for (i, f_at_x) in enumerate(f_at_x_samples)
		rand_gp_sample!(fx, f_at_x, xs, view(fs, i, :))
	end
	fs
end

# ╔═╡ 8064312b-bb7d-4d23-b6a8-3c089602a076
md"""
## Toy example setup
"""

# ╔═╡ 4e569b9f-48dd-40eb-8aa8-df4e84a81512
normalization_constant(f) = quadgk(f, -Inf, Inf)[1]

# ╔═╡ 4f1b79eb-d706-4495-904c-25c8db14c204
toy1d = let
	prior = Normal(0, 5.0)
	prior_pdf = f -> pdf(prior, f)
	y = true
	dist_y_given_f = f -> Bernoulli(logistic(f))
	lik_eval = f -> pdf(dist_y_given_f(f), y)
	un_post = f -> lik_eval(f) * prior_pdf(f)
	Z = normalization_constant(un_post)
	post = f -> un_post(f) / Z
	
	(; prior, prior_pdf, lik_eval, un_post, Z, post, y, dist_y_given_f)
end

# ╔═╡ dd7f234b-fc08-4939-a99b-0124286330ec
function normalization_constant2d(f)
	function int_f2(f1)
		return quadgk(f2 -> f(f1, f2), -60, 60)[1]
	end
	return quadgk(int_f2, -60, 60)[1]
end

# ╔═╡ 35ebcd54-a5bc-4a96-994a-81abcfaf3d37
toy2d = let
	k = 5^2*SqExponentialKernel()
	xs = [0., 1.2]
	prior = MvNormal(kernelmatrix(k, xs))
	y1 = true
	y2 = true
	ys = [y1, y2]

	prior_pdf = (f1,f2) -> pdf(prior, [f1,f2])
	dist_y_given_f = f -> Bernoulli(logistic(f))
	lik1 = f -> pdf(dist_y_given_f(f), y1)
	lik2 = f -> pdf(dist_y_given_f(f), y2)
	lik_eval = (f1,f2) -> lik1(f1) * lik2(f2)
	un_post = (f1,f2) -> lik_eval(f1,f2) * prior_pdf(f1,f2)
	
	Z = normalization_constant2d(un_post)
	post = (f1,f2) -> un_post(f1,f2) / Z
	(; prior, prior_pdf, lik_eval, un_post, Z, post, dist_y_given_f, ys, k, xs)
end

# ╔═╡ 25b4093b-9a3f-4fe0-9e48-d4ceab1dd08e
function plot_base(toy1d)
	plot(; xlims=(-20, 20), legend=:topleft, right_margin=8Plots.PlotMeasures.mm)
	plot!(toy1d.prior_pdf, label=raw"$p(f)$")
	
	lik_color = palette(:default)[2]
	colors = (; seriescolor=lik_color, foreground_color_axis=lik_color, foreground_color_border=lik_color, foreground_color_text=lik_color)
	plot!(twinx(), toy1d.lik_eval; label=raw"$p(y=1 | f)$", xticks=:none, colors...)
	
	plot!(toy1d.un_post, label=raw"$p(y=1 | f) p(f)$", seriescolor=4)
	plot!(toy1d.post, label=raw"$p(f | y=1)$", linestyle=:dash, seriescolor=3)
end

# ╔═╡ d16771ac-2017-4059-85e2-fc779be8f65c
md"""
# Gaussian posterior approximation
"""

# ╔═╡ fdfbcee6-74cf-47da-a7cb-19fdc3c5090f
md"""
prior $p(\mathbf{f}) = \mathcal{N}(\dots)$:
"""

# ╔═╡ a5ba1303-ae40-45e0-81e0-7e80757bf850
(collect(mean(toy2d.prior)), cov(toy2d.prior))

# ╔═╡ b67cb39b-159f-4583-918d-0ae6c2bb27dd
md"""
approximate posterior $q(\mathbf{f}) = \mathcal{N}(\dots)$:
"""

# ╔═╡ 532a8c7c-7d41-456a-ab90-a9f377fab506
begin
	bind_q_show = @bind q_show CheckBox()
	bind_q_m1 = @bind q_gp_m1 Slider(-10:0.5:10, default=0, show_value=true)
	bind_q_m2 = @bind q_gp_m2 Slider(-10:0.5:10, default=0, show_value=true)
	bind_q_std1 = @bind q_gp_std1 Slider(0:0.1:20,
		default=sqrt(var(toy2d.prior)[1]))
	bind_q_std2 = @bind q_gp_std2 Slider(0:0.1:20,
		default=sqrt(var(toy2d.prior)[2]))
	bind_q_corr = @bind q_gp_corr Slider(-1:0.1:1,
		default=cor(toy2d.prior)[1,2])
	
	md"""
$bind_q_show Show approximation

m₁ = $bind_q_m1
m₂ = $bind_q_m2

S₁₁ = $bind_q_std1
S₂₂ = $bind_q_std2
S₁₂ = $bind_q_corr
"""
end

# ╔═╡ ccf41bd0-98aa-4df8-b0fe-f9a9052c916b
qm, qS = let
	m = [q_gp_m1, q_gp_m2]
	S11 = q_gp_std1^2
	S22 = q_gp_std2^2
	corr = q_gp_corr
	S12 = sqrt(S11 * S22) * corr
	nugget = 1e-6
	S = [S11+nugget S12; S12 S22+nugget]
	m, S
end

# ╔═╡ f016fd91-6a3b-4dca-a571-0ecd82420a38
md"""
# Laplace
"""

# ╔═╡ 33b5cf2a-e25f-4b6f-af62-0d3748978015
md"""
## Taylor approximations in log space
"""

# ╔═╡ 2d038d36-14cb-47ea-8bfb-cc74b3f73fd1
h1d = log ∘ toy1d.un_post

# ╔═╡ 814b6f5b-a97a-4566-b971-a3f6d2a359df
@bind plot_laplace_x Slider(-10:0.1:15; default=0.3)

# ╔═╡ b1eead38-af18-487e-8687-1d6090cc8871
@bind plot_laplace_orders Slider(0:3, default=2)

# ╔═╡ 6ad4c38c-0150-43bb-aa58-60a9bc0ec0b5
md"""
## Laplace approximation
"""

# ╔═╡ 56f1057c-ffce-489c-aeec-65e7ad481ec1
deriv(f) = x -> ForwardDiff.derivative(f, float(x))

# ╔═╡ da766c97-3ab7-4e7e-b6c1-2d69afab5d98
h1d_derivs = let
	h = h1d
	h1 = deriv(h)
	h2 = deriv(h1)
	h3 = deriv(h2)
	
	[h1, h2, h3]
end

# ╔═╡ 576877b8-6429-493d-b1ce-7ae6c2454750
h1d_deriv_at_x, h1d_ts = let
	x0 = plot_laplace_x
	
	hx0 = h1d(x0)
	h1, h2, h3 = h1d_derivs
	h1x0 = h1(x0)
	h2x0 = h2(x0)
	h3x0 = h3(x0)
	
	t0 = _ -> hx0
	t1 = x -> t0(x) + h1x0 * (x - x0)
	t2 = x -> t1(x) + 0.5h2x0 * (x - x0)^2
	t3 = x -> t2(x) + h3x0 * (x - x0)^3 / 6
	
	(hx0, h1x0, h2x0, h3x0), (t0, t1, t2, t3)
end

# ╔═╡ bff6eeaf-87cb-45ec-9d71-4274ef0e4d04
let
	x0 = plot_laplace_x
	order = plot_laplace_orders
	
	exp_t2 = exp ∘ h1d_ts[3]
	cst, slope, curvature, _... = h1d_deriv_at_x
	σ² = - 1/curvature
	m = σ² * (slope - curvature * x0)
	qf = Normal(m, sqrt(σ²))

	plot(; xlims=(-20, 20), title="Laplace approximation", legend=:topleft)
	plot!(toy1d.un_post, color=:black, label="p(y|f) p(f)")
	plot!(toy1d.post, color=:black, label="p(f|y)", ls=:dash)
	scatter!([x0], [exp_t2(x0)], color=:black, label="x0")
	plot!(exp_t2, color=3, lw=2, label="exp(2nd-order Taylor)")
	plot!(qf, color=4, lw=2, label="q(f)")
end

# ╔═╡ f821c8a9-c584-4feb-b8c5-6331f984c1d8
let
	x0 = plot_laplace_x
	order = plot_laplace_orders
	
	h = h1d
	h1, h2, h3 = h1d_derivs
	
	xnew = x0 - h1(x0) / h2(x0)
	
	
	plot(; xlims=(-20, 20), ylims=(-20, 10), title="Taylor approximations to log-posterior", legend=:topleft)
	plot!(h, label="log p(y | f) + log p(f)", seriescolor=:black, linestyle=:dash)
	xgrid = range(xlims()...; length=100)
	scatter!([x0], [h(x0)], label=raw"$x_0$", seriescolor=:black)
	
	for i=0:length(h1d_ts)-1
		ith = Dict(0 => "0th", 1 => "1st", 2 => "2nd", 3 => "3rd")[i]
		order >= i && plot!(xgrid, h1d_ts[i+1], label="$(ith) order", seriescolor=i+1)
	end
	
	scatter!([xnew], [h(xnew)], label="Newton step", marker=:xcross, color=:black)
	
	plot!()
end

# ╔═╡ c4ccd718-da40-4ce8-99a1-35b90b8e2688
md"""
# KL between Gaussians
"""

# ╔═╡ 4b0e4f51-e961-4d53-a856-490430d7c9ab
begin
	q_mu_slider = @bind q_mu Slider(-10:0.1:10; default=0.0, show_value=true)
	q_sigma_slider = @bind q_sigma Slider(0.1:0.1:2; default=1.0, show_value=true)
	
	md"""
	 $p(f) = \mathcal{N}(0, 1)$
	
	 $q(f) = \mathcal{N}(\mu, \sigma^2)$

	 $\mu=$ $q_mu_slider
	
	 $\sigma=$ $q_sigma_slider
	"""
end

# ╔═╡ 7fa32949-adb5-44a9-afe8-1dfd946832ba
let
	prior = Normal()
	q = Normal(q_mu, q_sigma)
	kl_q_p = kldivergence(q, prior)
	kl_p_q = kldivergence(prior, q)
	plot(title="\$KL[q(f)\\|p(f)] = $(round(kl_q_p;digits=3)); KL[p(f)\\|q(f)] = $(round(kl_p_q;digits=3))\$", foreground_color_legend=nothing, xlims=(-10, 10))
	plot!(-10:0.1:10, prior, label="p(f)")
	plot!(-10:0.1:10, q, label="q(f)")
end

# ╔═╡ 361369ec-683a-4f69-b989-2875fc822ca1
md"""
## Multiplying Gaussians
"""

# ╔═╡ 194f5e7c-1df5-4f72-b013-0c7a2406cd5d
begin
	function mul_dist(a::NormalCanon, b::NormalCanon)
		# NormalCanon
		#     η::T       # σ^(-2) * μ
		#     λ::T       # σ^(-2)
		etaAmulB = a.η + b.η
		lambdaAmulB = a.λ + b.λ
		return NormalCanon(etaAmulB, lambdaAmulB)
	end

	mul_dist(a::Normal, b) = mul_dist(convert(NormalCanon, a), b)
	mul_dist(a, b::Normal) = mul_dist(a, convert(NormalCanon, b))
	mul_dist(a::Normal, b::Normal) = mul_dist(convert(NormalCanon, a), convert(NormalCanon, b))

	function mul_dist(a::MvNormalCanon, b::MvNormalCanon)
		# MvNormalCanon
		#    h::V    # potential vector, i.e. inv(Σ) * μ
		#    J::P    # precision matrix, i.e. inv(Σ)
		hAmulB = a.h + b.h
		JAmulB = a.J + b.J
		return MvNormalCanon(hAmulB, JAmulB)
	end

	mul_dist(a::MvNormal, b) = mul_dist(canonform(a), b)

	function div_dist(a::NormalCanon, b::NormalCanon)
		# NormalCanon
		#     η::T       # σ^(-2) * μ
		#     λ::T       # σ^(-2)
		etaAdivB = a.η - b.η
		lambdaAdivB = a.λ - b.λ
		return NormalCanon(etaAdivB, lambdaAdivB)
	end

	div_dist(a::Normal, b) = div_dist(convert(NormalCanon, a), b)
	div_dist(a, b::Normal) = div_dist(a, convert(NormalCanon, b))
	div_dist(a::Normal, b::Normal) = div_dist(convert(NormalCanon, a), convert(NormalCanon, b))
	
	mul_dist, div_dist
end

# ╔═╡ 533c3cf6-f639-4558-a1a1-9f72a0c4c4a6
let
	xs = -10:0.1:10
	n1 = Normal(-2, 2)
	n2 = Normal(3, 3)
	n3 = mul_dist(n1, n2)
	plot(xlims=extrema(xs))
	plot!(xs, n1, label="p(N₁)")# = p(N₃) / p(N₂)")
	plot!(xs, n2, label="p(N₂)")# = p(N₃) / p(N₁)")
	plot!(xs, n3, label="p(N₃) = p(N₁) p(N₂)", color=4)
end

# ╔═╡ f5b7c136-19d8-40aa-86ce-50132b6f767c
md"""
# Expectation Propagation in 2D
"""

# ╔═╡ d46f62fd-e24b-46a5-8ed6-31dc6415943e
begin
	bind_ep2d_init_m1 = @bind ep2d_init_m1 Slider(-5:0.1:5, default=0, show_value=true)
	bind_ep2d_init_m2 = @bind ep2d_init_m2 Slider(-5:0.1:5, default=0, show_value=true)
	bind_ep2d_init_s1 = @bind ep2d_init_s1 Slider(0.1:0.1:25, default=15, show_value=true)
	bind_ep2d_init_s2 = @bind ep2d_init_s2 Slider(0.1:0.1:25, default=15, show_value=true)
	:ep_2d_initialization_binds
end

# ╔═╡ 1db56946-2b37-4d11-b890-51479bd482d6
md"""
Site initialisation for EP 2D example

site 1: mean $bind_ep2d_init_m1, stdev $bind_ep2d_init_s1

site 2: mean $bind_ep2d_init_m2, stdev $bind_ep2d_init_s2
"""

# ╔═╡ 7cbaab0e-d49c-4ecb-b4f4-3a43ef43ab03
function ith_marginal(d::Union{MvNormal,MvNormalCanon}, i::Int)
    m = mean(d)
    v = var(d)
    return Normal(m[i], sqrt(v[i]))
end

# ╔═╡ 74e28e71-6ba6-439a-b1b1-174ebacbeaa6
function epsite_pdf(site, f)
    return site.Z * pdf(site.q, f)
end

# ╔═╡ 52622f99-1c6b-40f6-9fd2-36305510ea38
function moment_match(cav_i::UnivariateDistribution, lik_eval_i)
    lower = mean(cav_i) - 20 * std(cav_i)
    upper = mean(cav_i) + 20 * std(cav_i)
    m0, _ = quadgk(f -> pdf(cav_i, f) * lik_eval_i(f), lower, upper)
    m1, _ = quadgk(f -> f * pdf(cav_i, f) * lik_eval_i(f), lower, upper)
    m2, _ = quadgk(f -> f^2 * pdf(cav_i, f) * lik_eval_i(f), lower, upper)
    matched_Z = m0
    matched_mean = m1 / m0
    matched_var = m2 / m0 - matched_mean^2
    return (; Z=matched_Z, q=Normal(matched_mean, sqrt(matched_var)))
end

# ╔═╡ 73a3358e-911e-46a3-a9ad-5f53a904f8b9
function ep_approx_posterior(prior, sites::AbstractVector)
    canon_site_dists = [convert(NormalCanon, t.q) for t in sites]
    potentials = [q.η for q in canon_site_dists]
    precisions = [q.λ for q in canon_site_dists]
    ts_dist = MvNormalCanon(potentials, precisions)
    return mul_dist(prior, ts_dist)
end

# ╔═╡ e7fd80c5-003e-4fed-b2f1-2580bd24e751
q2, ep_res = let
	prior = toy2d.prior
	dist_y_given_f = toy2d.dist_y_given_f
	ys = toy2d.ys
	exact_post = toy2d.post
	
	N = length(ys)
	lik_evals = [f -> pdf(dist_y_given_f(f), y) for y in ys]
	# sites = [(; q=NormalCanon(0.0, floatmin(0.0))) for _=1:N]
	sites = [(; q=convert(NormalCanon, Normal(m, s))) for (m, s)
			in zip([ep2d_init_m1, ep2d_init_m2], [ep2d_init_s1, ep2d_init_s2])]
	
	q = ep_approx_posterior(prior, sites)
	
	colors = palette(:tab10)
	C1, C2, C3, C4, C5, _... = colors
	c_q = C1
	c_site = C2
	c_cavity = C3
	c_tilted = C4
	c_qhat = C5
	
	plts = []
	function storeplot!()
		push!(plts, deepcopy(plot!()))
	end
	
	flims = (-15, 15)
	plims = (-0.004, 0.135)
	fgrid = range(flims...; length=100)
	
	layout = @layout [a _
					  b{0.7w,0.7h} c]
	function plot1d!(i, fn; kwargs...)
		if i == 1
			plot!(fgrid, fn; subplot=1, kwargs...)
		elseif i == 2
			plot!(fn.(fgrid), fgrid; subplot=3, kwargs...)
		else
			error("i must be 1 or 2")
		end
	end
	plot1d!(i, fn::Distribution; kwargs...) = plot1d!(i, f -> pdf(fn, f); kwargs...)
	
	baseplot = plot(; size=(600, 600), legend=:topleft, link=:both, layout=deepcopy(layout),
		foreground_color_legend=nothing, background_color_legend=nothing)
	plot!(subplot=2, xlim=flims, ylim=flims, xlabel="f₁", ylabel="f₂")#, aspect_ratio=:auto)
	plot!(subplot=1, xlim=flims, ylim=plims, yticks=[0.0, 0.05, 0.1])
	plot!(subplot=3, xlim=plims, ylim=flims, xticks=[0.0, 0.05, 0.1], legend=:bottomright)

	t_idx = ["₁", "₂"]
	marginals = [[ith_marginal(q, i)] for i=1:N]

	n_steps = 3
	for (step, i) in enumerate(repeat(1:N, n_steps))
		q_fi = ith_marginal(q, i)
		cav_i = div_dist(q_fi, sites[i].q)
		qhat_i = moment_match(cav_i, lik_evals[i])
		new_t = div_dist(qhat_i.q, cav_i)
		new_sites = deepcopy(sites)
		new_sites[i] = (; q=new_t)
		new_q = ep_approx_posterior(prior, new_sites)

		plot!(deepcopy(baseplot))
		plot!(subplot=1, title="step $(Int(ceil(step/2))), site $i")
		
		contour!(fgrid, fgrid, exact_post, subplot=2, colorbar=nothing, ls=:dash)
		contour!(fgrid, fgrid, (f1, f2) -> pdf(prior, [f1, f2]), subplot=2, color=:black, ls=:dot)
		step == 1 && storeplot!()
		
		contour!(fgrid, fgrid, (f1, f2) -> pdf(q, [f1, f2]), subplot=2)
		step == 1 && storeplot!()
		
		for k=1:N
			sub_k = t_idx[k]
			plot1d!(k, ith_marginal(q, k), label="initial q(f$(sub_k))", color=c_q)
			plot1d!(k, sites[k].q, label="initial site: t$(sub_k)(f$(sub_k))", color=c_site, ls=:dash, lw=2)
		end
		storeplot!()
		
		sub_i = t_idx[i]
		plot1d!(i, cav_i, label="cavity q₋$(sub_i)(f$(sub_i))", color=c_cavity, ls=:dash, lw=2)
		storeplot!()
		
		# plot!(fgrid, un_post, label="p(f) p(y | f)", color=:black, lw=2)
		plot1d!(i, f -> pdf(cav_i, f) * lik_evals[i](f), label="tilted q`$(sub_i)(f$(sub_i))", color=c_tilted, ls=:dash, lw=2)
		storeplot!()
		
		plot1d!(i, f -> epsite_pdf(qhat_i, f), label="matched q̂", color=c_qhat, lw=2, ls=:dash)
		storeplot!()
		
		plot1d!(i, qhat_i.q, label=" — normalized", color=c_qhat, lw=2)
		storeplot!()
		
		plot1d!(i, new_t, label="new site: t$(sub_i)'(f$(sub_i))", color=c_site, lw=2)
		storeplot!()
		
		contour!(fgrid, fgrid, (f1, f2) -> pdf(new_q, [f1, f2]), subplot=2)
		storeplot!()
		
		for k=1:N
			sub_k = t_idx[k]
			q_k = ith_marginal(new_q, k)
			plot1d!(k, q_k, label="new q'(f$(sub_k))", color=c_q, ls=:dot, lw=2)
			push!(marginals[k], q_k)
		end
		storeplot!()
		
		q = new_q
		sites = new_sites
	end
	
	plot!(deepcopy(baseplot))
	plot!(subplot=1, title="converged")

	contour!(fgrid, fgrid, exact_post, subplot=2, colorbar=nothing, ls=:dash)
	contour!(fgrid, fgrid, (f1, f2) -> pdf(q, [f1, f2]), subplot=2)

	post_marg = (f1 -> quadgk(f2 -> toy2d.post(f1, f2), flims...)[1]).(fgrid) 
	plot!(fgrid, post_marg, label="exact posterior", lw=2, ls=:dash, color=:black, subplot=1)
	plot!(post_marg, fgrid, label="exact posterior", lw=2, ls=:dash, color=:black, subplot=3)

	for k=1:N
		sub_k = t_idx[k]
		plot1d!(k, ith_marginal(q, k), label="q(f$(sub_k))", color=c_q, lw=2)
		# plot1d!(k, sites[k].q, label="t$(sub_k)(f$(sub_k))", color=c_site, ls=:dash, lw=2)
	end
	storeplot!()
	
	plts, (; fgrid, post_marg, q)
end

# ╔═╡ fea215a4-0f53-4262-b6cd-b67a648344a5
@bind q2_idx Slider(1:length(q2), default=2)#default=length(q2))

# ╔═╡ 32ef190d-21eb-4e7d-bb68-a171007409a7
q2[q2_idx]

# ╔═╡ e93580fc-7666-4548-bba9-2f2a3258c24e
md"""
# MCMC (Metropolis–Hastings)
"""

# ╔═╡ 5dc0aaec-31b7-41c1-8ca3-5dc8de8c9b20
function simple_mh!(un_post_fn, f_init, n_steps; proposal_scale=1.0)
	f_dim = length(f_init)
	fs = zeros(f_dim, n_steps)
	fs[:, 1] .= f_init
	proposals = zeros(f_dim, n_steps - 1)
	n_total = 0
	n_accepted = 0
	for i = 2:n_steps
		f_last = fs[:, i - 1]
		f_proposal = rand(MvNormal(f_last, proposal_scale))
		proposals[:, i - 1] = f_proposal
		q0 = un_post_fn(f_last...)
		q1 = un_post_fn(f_proposal...)
		if rand() < q1 / q0
			fs[:, i] = f_proposal
			n_accepted += 1
		else
			fs[:, i] = f_last
		end
		n_total += 1
	end
	fs, proposals, n_accepted // n_total
end

# ╔═╡ 39c86e0d-7c2f-496d-80dd-5395b85e7e68
md"""
## 1D example
"""

# ╔═╡ 8da12d3e-81a4-4a88-b2a9-624e66766694
let
	prior = toy1d.prior
	un_post = toy1d.un_post
	post = toy1d.post
	
	n_steps = 100
	fs, proposals, ar = simple_mh!(un_post, rand(prior), n_steps; proposal_scale=4)
	
	accepted = isapprox.(proposals, fs[:,2:end])
	rejected_fs = []
	accepted_fs = []
	colors = []
	for (f, p, a) in zip(fs[:,2:end], proposals, accepted)
		if a
			push!(accepted_fs, p)
			push!(colors, 3)
		else
			push!(accepted_fs, f)
			push!(colors, 4)
			push!(rejected_fs, p)
		end
	end
	
	plt1 = plot(ylim=(-15,15))
	plot!(plt1, fs', label="samples")
	# scatter!(plt1, 2:n_steps, vec(proposals); color=color', label="proposals")
	scatter!(plt1, 2:n_steps, rejected_fs; color=2, label="proposals")
	scatter!(plt1, 2:n_steps, accepted_fs, color=colors, label="")
	annotate!(plt1, 4, -13.5, text("acceptance ratio $(round(100.0ar, digits=1)) %", 8, :left))
	
	plt2 = plot()
	plot!(plt2, -10:0.05:20, post, label="exact posterior")
	vline!(plt2, fs', alpha=0.3, label="samples")
	histogram!(plt2, fs', alpha=0.3, normalize=true, label="normalized histogram")
	
	plot(plt1, plt2)
end

# ╔═╡ 27d1ad70-3440-4c22-839c-57e785908441
md"""
## 2D example
"""

# ╔═╡ 77f711b0-0bf5-4aac-ae83-03c430348317
function run_mcmc(toy2d; n_steps=10000, proposal_scale=2.0)
	prior = toy2d.prior
	un_post = toy2d.un_post
	
	flims = (-15, 15)
	fgrid = range(flims...; length=70)
	p_base = plot(size=(500, 500), aspect_ratio=1, xlims=flims, ylims=flims, legend=:bottomleft, foreground_color_legend=nothing, background_color_legend=nothing)
	# contour!(fgrid, fgrid, prior_pdf)
	contour!(fgrid, fgrid, un_post, colorbar=false)
	# contour!(fgrid, fgrid, post)
	
	fs, proposals, ar = simple_mh!(un_post, rand(prior), n_steps; proposal_scale)
	
	function plot_proposal_dist!(i)
		proposal_dist = MvNormal(fs[:, i], proposal_scale)
		covellipse2!(mean(proposal_dist), cov(proposal_dist); seriescolor=1, alpha=0.1, label="proposal distribution")
	end
	
	function plot_proposal_arrow!(i; label="")
		plot!([fs[1, i], proposals[1, i]], [fs[2, i], proposals[2, i]]; arrow=arrow(:closed), label, seriescolor=4)
	end
	
	function plot_proposal_result!(i)
		accepted = proposals[:, i] == fs[:, i+1]
		proposal_color = accepted ? 3 : 2
		scatter!(proposals[1,i:i], proposals[2,i:i], seriescolor=proposal_color, label="new proposal")
		return accepted
	end
	
	plts = [p_base]

	plot!(deepcopy(p_base))
	scatter!(fs[1,1:1], fs[2,1:1], seriescolor=4, label="initial state")
	
	push!(plts, deepcopy(plot!()))
	
	plot_proposal_dist!(1)
	push!(plts, deepcopy(plot!()))

	plt = deepcopy(plot!())
	
	plot_proposal_arrow!(1, label="new proposal")
	push!(plts, deepcopy(plot!()))
	
	plot!(plt)
	annotate!(-14, 14, text("step 1", pointsize=10, halign=:left))
	plot_proposal_arrow!(1)
	plot_proposal_result!(1)
	push!(plts, plot!())

	n_total = n_accepted = 0
	for i=2:min(n_steps-1, 200)
		plt = plot!(deepcopy(p_base))
		scatter!(fs[1,1:i-1], fs[2,1:i-1]; seriescolor=1, alpha=0.3, label="previous states", markersize=2)
		scatter!(fs[1,i:i], fs[2,i:i], seriescolor=4, label="last state")
		
		plot_proposal_dist!(i)
		plot_proposal_arrow!(i)
		accepted = plot_proposal_result!(i)
		
		n_accepted += accepted
		n_total += 1
		ar = convert(Int, round(n_accepted / n_total * 100))
		annotate!(-14, 14, text("step $i", pointsize=10, halign=:left))
		annotate!(-14, 12, text("acceptance ratio so far = $ar %", pointsize=8, halign=:left))

		push!(plts, plt)
	end
	return plts, ar, fs
end

# ╔═╡ 52bfff38-9bad-4ec8-bccd-decbfd46525c
begin
	Random.seed!(4382) # reasonably good example for burn-in
	mcmc_plts, mcmc_acceptance_ratio, mcmc_fs = run_mcmc(
		toy2d;
		# n_steps=10000,
		# proposal_scale=2.0,
		# n_steps=500,
	)
end;

# ╔═╡ 26b7c857-2d6d-4bd6-b3fd-c9ed50aa9223
let
	flims = (-15, 15)
	fgrid = range(flims...; length=60)
	xgrid = range(-3, 3; length=60)
	
	baseplot() = plot(; size=(600, 500), aspect_ratio=1, xlims=flims, ylims=flims)
	
	p1 = baseplot()
	plot!(xlabel="f(x₁)", ylabel="f(x₂)")
	clim = (0, 0.02)
	
	contour!(fgrid, fgrid, toy2d.post; label="exact posterior", linestyle=:dash, clim,
		colorbar=nothing)
	
	q_show && covellipse2!(qm, qS, color=1)

	f = GP(toy2d.k)
	fx = f(toy2d.xs)
	
	seed = 123
	fpostMC = mcmc_fs[:, 2000:200:end] # [:, 1000:50:end]
	Random.seed!(seed)
	fs = get_gp_samples(xgrid, fx, eachcol(fpostMC))
	
	p2 = plot(; xlim=extrema(xgrid), xlabel="x", ylabel="p(f | y)")
	plot!(xgrid, logistic.(fs'), color=3, label="", alpha=0.3)
	scatter!(toy2d.xs, toy2d.ys, color=2, label="")
	vline!(toy2d.xs, color=2, label="", ls=:dash)
	
	f = posterior(SVGP(f(toy2d.xs), MvNormal(qm, qS)))
	Random.seed!(seed)
	fs_q = rand(f(xgrid, 1e-8), size(fpostMC, 2))
	
	p3 = if q_show
		plot(; xlim=extrema(xgrid), xlabel="x", ylabel="p(f | u) q(u)")
		plot!(xgrid, logistic.(fs_q), color=1, label="", alpha=0.3)
	else
		p_blank
	end
	
	plot(p2, p1, p3)
end

# ╔═╡ de3c14d3-069e-4a4e-88a8-1869d8a2450d
md"""
### MCMC visualization
"""

# ╔═╡ 561154af-daf8-41ed-9ee6-c379bbe00c6e
@bind plot_time Slider(1:length(mcmc_plts))

# ╔═╡ 33e630bd-6825-47b2-917c-f92d1f49e308
mcmc_plts[plot_time]

# ╔═╡ afc478d7-58f2-4acd-ab3e-492e4414467c
md"""
Final MCMC acceptance ratio: $mcmc_acceptance_ratio %
"""

# ╔═╡ 000cbb3e-abe4-429c-a3c2-c132bbf20f38
md"""
### series and autocorrelation
"""

# ╔═╡ 9f6db10d-3762-4936-a4b5-8d07a6de316e
let
	p1 = plot(xlabel="step", legend=:bottomright)
	plot!(mcmc_fs', label=[raw"$f_1$" raw"$f_2$"], ylabel=raw"$f$")
	p2 = plot(autocor(mcmc_fs'), ylabel="autocorrelation", label="", ylim=(0, 1))
	plot(p1, p2, layout=@layout [a{0.7w} b])
end

# ╔═╡ bc97d8dc-14af-42e7-afde-04dc090f6d69
md"""
### final posterior plots
"""

# ╔═╡ 5cb69ff5-73c9-4ad3-8101-4bee33c342eb
mcmc_final_plots = let
	flims = (-15, 15)
	fgrid = range(flims...; length=70)
	
	baseplot() = plot(; size=(600, 500), aspect_ratio=1, xlims=flims, ylims=flims)
	
	p0 = baseplot()
	contour!(fgrid, fgrid, toy2d.un_post, title="unnormalized posterior")
	
	p1a = plot(deepcopy(p0))
	scatter!(mcmc_fs[1, :], mcmc_fs[2, :], alpha=0.1, label="", markerstrokewidth=0.1,
		title="MCMC samples"
	)

	p1b = baseplot()
	histogram2d!(mcmc_fs[1, :], mcmc_fs[2, :], bins=30, normalized=true,
		title="histogram")
	
	p2 = baseplot()
	contour!(fgrid, fgrid, toy2d.post, title="exact posterior")
	
	(p0, p1a, p1b, p2)
end

# ╔═╡ 5f90502c-529b-4abe-9eec-07e780b82ec8
@bind plot_mcmc_final Slider(1:length(mcmc_final_plots))

# ╔═╡ e7b25f12-54de-448b-a1ca-2b8996f51f49
mcmc_final_plots[plot_mcmc_final]

# ╔═╡ e751701a-7b99-48fa-b9fb-aad4eb9294b7
md"""
# Bonus: approximate posterior predictions
"""

# ╔═╡ 72319b85-874e-440c-852e-ee5b3e76d2ca
begin
	bind_x1 = @bind vi_gp_xobs1 Scrubbable(-8:0.1:8, default=-2)
	bind_x2 = @bind vi_gp_xobs2 Scrubbable(-8:0.1:8, default=2)
	bind_y1 = @bind vi_gp_yobs1 Scrubbable(-2:0.1:2)
	bind_y2 = @bind vi_gp_yobs2 Scrubbable(-2:0.1:2)
	bind_std1 = @bind vi_gp_std1 Slider(0:0.1:4.5, default=2)
	bind_std2 = @bind vi_gp_std2 Slider(0:0.1:4.5, default=2)
	bind_corr = @bind vi_gp_corr Slider(-1:0.1:1, default=0.4)
	
	md"""
x₁ = $bind_x1
y₁ = $bind_y1

x₂ = $bind_x2
y₂ = $bind_y2

S₁₁ = $bind_std1
S₂₂ = $bind_std2
S₁₂ = $bind_corr
"""
end

# ╔═╡ 0beb022f-a9d6-41ee-afe5-5448445b5d12
begin
	xobs = Float64[vi_gp_xobs1, vi_gp_xobs2]
	yobs = Float64[vi_gp_yobs1, vi_gp_yobs2]
end;

# ╔═╡ b26e7cfd-40bc-4d90-bf95-aeb298ce6e93
S = let
	S11 = vi_gp_std1^2
	S22 = vi_gp_std2^2
	corr = vi_gp_corr
	S12 = sqrt(S11 * S22) * corr
	nugget = 1e-6
	[S11+nugget S12; S12 S22+nugget]
end

# ╔═╡ be4cf1b2-c170-447e-a445-46284b0a0c4a
let
	k = 4with_lengthscale(SqExponentialKernel(), 3)
	fprior = GP(k)
	f = posterior(SVGP(fprior(xobs), MvNormal(yobs, S)))
	K = cov(fprior(xobs))
	# Sigma = inv(inv(S) - inv(K))
	# f = posterior(fprior(xobs, Sigma), yobs)
	m, C = mean_and_cov(f(xobs))
	xgrid = -10:.1:10
	fgrid = -5:.1:5
	
	p1 = plot(; xlim=extrema(xgrid), ylim=extrema(fgrid))
	Random.seed!(123)
	plot!(xgrid, f, ribbon_scale=2, label="", color=1)
	plot!(xgrid, f, ribbon_scale=1, label="", color=1)
	sampleplot!(xgrid, f, samples=20, color=:blue)
	scatter!(xobs, m, yerror=sqrt.(diag(C)), color=3, label="")
	
	p2 = plot(; xlim=extrema(fgrid), ylim=extrema(fgrid), aspect_ratio=1, legend=:bottomright)
	# contour!(fgrid, fgrid, (f1, f2) -> pdf(MvNormal(yobs, S), [f1, f2]))
	covellipse2!(zeros(2), K, label="p(f₁, f₂)", color=2)
	# covellipse2!(yobs, S, label="S", color=1)
	covellipse2!(m, C, label="q(f₁, f₂)", color=1)
	scatter!(yobs[1:1], yobs[2:2], label="", color=3)
	plot(p1, p2, size=(700,300), layout=@layout [a{0.6w} b])
end

# ╔═╡ Cell order:
# ╟─3a6a35fd-9acd-49db-9249-d8850fa1e275
# ╠═c9603eb0-0407-11ec-1751-3de72d9fd996
# ╠═52074a94-620e-417d-afc2-e9c05024069d
# ╟─0b7b7155-1bef-4070-81d1-b3a506d7fb7e
# ╟─4e22cdca-5180-4875-a43d-16efa21fc274
# ╠═507d6b85-92ab-458c-a6f1-46a5f9b178be
# ╟─9ba94fc8-1951-4c21-9c3a-e655a70321e7
# ╟─57cc2052-72c1-484a-88be-b3b216ff045b
# ╟─c4825dde-ddf9-4a16-a9ea-c65a8eb50319
# ╟─8064312b-bb7d-4d23-b6a8-3c089602a076
# ╠═4e569b9f-48dd-40eb-8aa8-df4e84a81512
# ╟─4f1b79eb-d706-4495-904c-25c8db14c204
# ╟─dd7f234b-fc08-4939-a99b-0124286330ec
# ╟─35ebcd54-a5bc-4a96-994a-81abcfaf3d37
# ╟─25b4093b-9a3f-4fe0-9e48-d4ceab1dd08e
# ╟─d16771ac-2017-4059-85e2-fc779be8f65c
# ╟─fdfbcee6-74cf-47da-a7cb-19fdc3c5090f
# ╟─a5ba1303-ae40-45e0-81e0-7e80757bf850
# ╟─b67cb39b-159f-4583-918d-0ae6c2bb27dd
# ╟─ccf41bd0-98aa-4df8-b0fe-f9a9052c916b
# ╟─532a8c7c-7d41-456a-ab90-a9f377fab506
# ╟─26b7c857-2d6d-4bd6-b3fd-c9ed50aa9223
# ╟─f016fd91-6a3b-4dca-a571-0ecd82420a38
# ╟─33b5cf2a-e25f-4b6f-af62-0d3748978015
# ╟─2d038d36-14cb-47ea-8bfb-cc74b3f73fd1
# ╟─da766c97-3ab7-4e7e-b6c1-2d69afab5d98
# ╟─576877b8-6429-493d-b1ce-7ae6c2454750
# ╟─f821c8a9-c584-4feb-b8c5-6331f984c1d8
# ╠═814b6f5b-a97a-4566-b971-a3f6d2a359df
# ╠═b1eead38-af18-487e-8687-1d6090cc8871
# ╟─6ad4c38c-0150-43bb-aa58-60a9bc0ec0b5
# ╟─bff6eeaf-87cb-45ec-9d71-4274ef0e4d04
# ╟─56f1057c-ffce-489c-aeec-65e7ad481ec1
# ╟─c4ccd718-da40-4ce8-99a1-35b90b8e2688
# ╟─4b0e4f51-e961-4d53-a856-490430d7c9ab
# ╟─7fa32949-adb5-44a9-afe8-1dfd946832ba
# ╟─361369ec-683a-4f69-b989-2875fc822ca1
# ╟─194f5e7c-1df5-4f72-b013-0c7a2406cd5d
# ╟─533c3cf6-f639-4558-a1a1-9f72a0c4c4a6
# ╟─f5b7c136-19d8-40aa-86ce-50132b6f767c
# ╟─d46f62fd-e24b-46a5-8ed6-31dc6415943e
# ╟─1db56946-2b37-4d11-b890-51479bd482d6
# ╟─7cbaab0e-d49c-4ecb-b4f4-3a43ef43ab03
# ╟─74e28e71-6ba6-439a-b1b1-174ebacbeaa6
# ╟─52622f99-1c6b-40f6-9fd2-36305510ea38
# ╟─73a3358e-911e-46a3-a9ad-5f53a904f8b9
# ╠═e7fd80c5-003e-4fed-b2f1-2580bd24e751
# ╟─32ef190d-21eb-4e7d-bb68-a171007409a7
# ╟─fea215a4-0f53-4262-b6cd-b67a648344a5
# ╟─e93580fc-7666-4548-bba9-2f2a3258c24e
# ╠═5dc0aaec-31b7-41c1-8ca3-5dc8de8c9b20
# ╟─39c86e0d-7c2f-496d-80dd-5395b85e7e68
# ╟─8da12d3e-81a4-4a88-b2a9-624e66766694
# ╟─27d1ad70-3440-4c22-839c-57e785908441
# ╟─77f711b0-0bf5-4aac-ae83-03c430348317
# ╠═52bfff38-9bad-4ec8-bccd-decbfd46525c
# ╟─de3c14d3-069e-4a4e-88a8-1869d8a2450d
# ╟─33e630bd-6825-47b2-917c-f92d1f49e308
# ╟─561154af-daf8-41ed-9ee6-c379bbe00c6e
# ╟─afc478d7-58f2-4acd-ab3e-492e4414467c
# ╟─000cbb3e-abe4-429c-a3c2-c132bbf20f38
# ╟─9f6db10d-3762-4936-a4b5-8d07a6de316e
# ╟─bc97d8dc-14af-42e7-afde-04dc090f6d69
# ╟─5cb69ff5-73c9-4ad3-8101-4bee33c342eb
# ╟─e7b25f12-54de-448b-a1ca-2b8996f51f49
# ╟─5f90502c-529b-4abe-9eec-07e780b82ec8
# ╟─e751701a-7b99-48fa-b9fb-aad4eb9294b7
# ╟─0beb022f-a9d6-41ee-afe5-5448445b5d12
# ╟─72319b85-874e-440c-852e-ee5b3e76d2ca
# ╟─b26e7cfd-40bc-4d90-bf95-aeb298ce6e93
# ╟─be4cf1b2-c170-447e-a445-46284b0a0c4a
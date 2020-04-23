#####Julia Packages#####
using CSV, DataFrames
using JuMP
using Gurobi
#using Plots
#using StatsPlots
#using Plotly
#using PyPlot



#####Input Data#####
include("input.jl")

p2pdata = load_data("data/Residential_data_raw.csv")

n_peers = p2pdata.n_peers
n_timesteps = p2pdata.n_timesteps
peers = p2pdata.peers

feedin_tariff = 0.12 / 1000 # 12 cent per kWh
grid_tariff = 0.3 / 1000 # 30 cent per kWh

#c_range = (0.151, 0.242) ./ 1000  #Min Max cost of DER
#lcoe = c_range[1] .+ rand(n_peers) .* (c_range[2] - c_range[1])
lcoe = [0.00022642060704422984, 0.00015459726304964378, 0.00023073075451998822, 0.00016629819870396203, 0.0001784348070200074, 0.0002056126965244984, 0.00015672053527754508, 0.00015105311223707551, 0.0002286007973190153, 0.0002268097960918653, 0.0001630413912447715, 0.00021740187867045442, 0.0001806392010936946, 0.0002263521491564626, 0.0001796451977605077, 0.0001902877484560514, 0.00015942587498472369, 0.00015145981432357808, 0.000179100701558622, 0.00017687811364570179, 0.00019948525514998768, 0.00019225041152258238, 0.00015657525154546566, 0.00015202344929618163, 0.0001693428002416519, 0.00016134313307556846, 0.00019053739155264101, 0.00017988916385850993, 0.0001559425211674564]
lcoe_sum = sum(lcoe[i] for i in 1:n_peers)

v = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps, i in 1:n_peers
    v[t, i] =  peers[i].test[t]
end

d_max = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        d_max[t, i] = peers[i].dmax[t]
    end
end

p = v .* 0.0000001
o = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        o[t, i] = p[i] * peers[i].dmax[t]
    end
end

max_gen_hourly = [sum(peers[i].gmax[t] for i in 1:n_peers) for t in 1:n_timesteps]
max_dem_hourly = [sum(peers[i].dmax[t] for i in 1:n_peers) for t in 1:n_timesteps]

lcoe_all = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps, i in 1:n_peers
    lcoe_all[t, i] = lcoe[i]
end



#####Gross-Optimization: With Demand and Surplus#####
m = Model(with_optimizer(Gurobi.Optimizer))

@variable(m, g[1:n_timesteps, 1:n_peers])
@variable(m, d[1:n_timesteps, 1:n_peers])
@variable(m, deficit[1:n_timesteps])
@variable(m, surplus[1:n_timesteps])

for t in 1:n_timesteps
    if max_gen_hourly[t] >= max_dem_hourly[t]
        @constraint(m, deficit[t] == 0)
    else
        @constraint(m, surplus[t] == 0)
    end
end

@constraint(m, λ[t = 1:n_timesteps], sum(d[t, i] for i in 1:n_peers) + surplus[t] == sum(g[t, i] for i in 1:n_peers) + deficit[t])

@constraint(m, δgp[t = 1:n_timesteps, i = 1:n_peers], g[t, i] <= peers[i].gmax[t])
@constraint(m, δgm[t = 1:n_timesteps, i = 1:n_peers], g[t, i] >= peers[i].gmin[t])

@constraint(m, δdp[t = 1:n_timesteps, i = 1:n_peers], d[t, i] <= peers[i].dmax[t])
@constraint(m, δdm[t = 1:n_timesteps, i = 1:n_peers], d[t, i] >= peers[i].dmin[t])

@constraint(m, [t = 1:n_timesteps], surplus[t] >= 0)
@constraint(m, [t = 1:n_timesteps], deficit[t] >= 0)

a = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        a[t, i] = -p[i]
    end
end
b = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        b[t, i] = 2 * p[i] * peers[i].dmax[t]
    end
end
c = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        c[t, i] = o[t, i] * peers[i].dmax[t] - p[i] * peers[i].dmax[t]^2
    end
end

generation_costs_total = [sum(lcoe[i] * g[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
demand_utility_total = [sum(a[t, i] * d[t, i]^2 + b[t, i] * d[t, i] + c[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
@expression(m, deficit_costs[t = 1:n_timesteps], grid_tariff * deficit[t])
@expression(m, surplus_revenue[t = 1:n_timesteps], feedin_tariff * surplus[t])

@objective(m, Min, sum(generation_costs_total[t] + deficit_costs[t] - demand_utility_total[t] - surplus_revenue[t] for t in 1:n_timesteps))

optimize!(m)
status = termination_status(m)

obv = objective_value(m)



#####Net-Optimization: Without Deficit and Surplus#####
net = Model(with_optimizer(Gurobi.Optimizer))

@variable(net, net_g[1:n_timesteps, 1:n_peers]) #generation
@variable(net, net_d[1:n_timesteps, 1:n_peers]) #demand

@constraint(net, [t = 1:n_timesteps], sum(net_d[t, i] for i in 1:n_peers) == sum(net_g[t, i] for i in 1:n_peers))

@constraint(net, [t = 1:n_timesteps, i = 1:n_peers], net_g[t, i] <= peers[i].gmax[t])
@constraint(net, [t = 1:n_timesteps, i = 1:n_peers], net_g[t, i] >= peers[i].gmin[t])

@constraint(net, [t = 1:n_timesteps, i = 1:n_peers], net_d[t, i] <= peers[i].dmax[t])
@constraint(net, [t = 1:n_timesteps, i = 1:n_peers], net_d[t, i] >= peers[i].dmin[t])

net_generation_costs_total = [sum(lcoe[i] * net_g[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
net_demand_utility_total = [sum(a[t, i] * net_d[t, i]^2 + b[t, i] * net_d[t, i] + c[t, i] for i in 1:n_peers) for t in 1:n_timesteps]

@objective(net, Min, sum(net_generation_costs_total[t] - net_demand_utility_total[t] for t in 1:n_timesteps))

optimize!(net)
status = termination_status(net)

obv = objective_value(net)



#####Results general#####
d_results_individual = value.(d)
g_results_individual = value.(g)
d_results_total_hourly = [sum(d_results_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
g_results_total_hourly = [sum(g_results_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
d_results_total = sum(d_results_total_hourly[t] for t in 1:n_timesteps)/1000
g_results_total = sum(g_results_total_hourly[t] for t in 1:n_timesteps)/1000

net_d_results_individual = value.(net_d) #consumption without deficit option
net_g_results_individual = value.(net_g) #generation without surplus option
net_d_results_total_hourly = [sum(net_d_results_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps] #consumption without deficit option
net_g_results_total_hourly = [sum(net_g_results_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps] #generation without surplus option

marginal_price_sellers_unbereinigt = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        marginal_price_sellers_unbereinigt[t, i] = lcoe[i]
    end
end
marginal_price_sellers_bereinigt = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        if g_results_individual[t, i] > 1e-5
        marginal_price_sellers_bereinigt[t, i] = lcoe[i]
        else
        marginal_price_sellers_bereinigt[t, i] = 0
        end
    end
end
generation_costs_individual_results = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        generation_costs_individual_results[t, i] = lcoe[i] * g_results_individual[t, i]
    end
end
generation_costs_total_results = value.(generation_costs_total)

marginal_utility_buyers_unbereinigt = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        marginal_utility_buyers_unbereinigt[t, i] = 2 * a[t, i] * d_results_individual[t, i] + b[t, i]
    end
end

marginal_utility_buyers_bereinigt = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        if net_d_results_individual[t, i] > 1e-5
        marginal_utility_buyers_bereinigt[t, i] = 2 * a[t, i] * d_results_individual[t, i] + b[t, i]
        else
        marginal_utility_buyers_bereinigt[t, i] = 0
        end
    end
end
marginal_utility_buyers_bereinigt_max = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        if d_results_individual[t, i] > 1e-5
        marginal_utility_buyers_bereinigt_max[t, i] = a[t, i] * d_max[t, i]^2 + b[t, i] * d_max[t, i] + c[t, i]
        else
        marginal_utility_buyers_bereinigt_max[t, i] = 0
        end
    end
end

demand_utility_individual_results = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        demand_utility_individual_results[t, i] = a[t, i] * d_results_individual[t, i]^2 + b[t, i] * d_results_individual[t, i] + c[t, i]
    end
end
demand_utility_total_results = value.(demand_utility_total)

surplus_results = max_gen_hourly - g_results_total_hourly
surplus_results_total = sum(surplus_results[t] for t in 1:n_timesteps)/1000
surplus_revenue_results = surplus_results * feedin_tariff
surplus_revenue_total = sum(surplus_revenue_results[t] for t in 1:n_timesteps)

deficit_results = value.(deficit)
deficit_results_total = sum(deficit_results[t] for t in 1:n_timesteps)/1000
deficit_costs_results = value.(deficit_costs)
deficit_costs_total = sum(deficit_costs_results[t] for t in 1:n_timesteps)

NetGenSurplus = surplus_results - deficit_results

deficit_consumption_individual = d_results_individual - net_d_results_individual #Individual consumption from grid
surplus_generation_individual = g_results_individual - net_g_results_individual #Individual feedin into grid

Sellers = n_peers
Buyers = n_peers
N = Sellers + Buyers
Q = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        Q[t, i] = net_d_results_total_hourly[t] + net_g_results_total_hourly[t]
    end
end
generating_sellers = zeros(n_timesteps)
for t in 1:n_timesteps, i in 1:n_peers
    if g_results_individual[t, i] > 0
    generating_sellers[t] = generating_sellers[t] + 1
    end
end
consuming_buyers = zeros(n_timesteps)
for t in 1:n_timesteps, i in 1:n_peers
    if d_results_individual[t, i] > 0
    consuming_buyers[t] = consuming_buyers[t] + 1
    end
end
bpt = 1 #boxplot timestep

sortingCostl2h = marginal_price_sellers_bereinigt
presorted_marginal_cost_l2h = sort(sortingCostl2h, dims = 2, rev = false)
sortingCosth2l = marginal_price_sellers_bereinigt
presorted_marginal_cost_h2l = sort(sortingCosth2l, dims = 2, rev = true)
lcoe_boxplot = []
lcoe_boxplot_rev = []
    for i in 1:n_peers
    if presorted_marginal_cost_l2h[bpt, i] > 0
        push!(lcoe_boxplot, presorted_marginal_cost_l2h[bpt, i])
    else
    end
    if presorted_marginal_cost_h2l[bpt, i] > 0
    push!(lcoe_boxplot_rev, presorted_marginal_cost_h2l[bpt, i])
else
end
end

lcoe_g_abgleich = zeros(n_timesteps, n_peers)
for i in 1:n_peers
    for t in 1:n_timesteps
        lcoe_g_abgleich[t, i] = g_results_individual[t, i] * lcoe_all[t, i]
end
end

g_presorted = []
g_presorted_rev = []
for i in 1:n_peers
    for j in 1:n_peers
        if presorted_marginal_cost_l2h[bpt, i] > 0
        if presorted_marginal_cost_l2h[bpt, i] * g_results_individual[bpt, j] == lcoe_g_abgleich[bpt, j]
            push!(g_presorted, g_results_individual[bpt, j])
        end
        end
        if presorted_marginal_cost_h2l[bpt, i] > 0
        if presorted_marginal_cost_h2l[bpt, i] * g_results_individual[bpt, j] == lcoe_g_abgleich[bpt, j]
            push!(g_presorted_rev, g_results_individual[bpt, j])
        end
        end
    end
end
n_peers_neu = size(g_presorted, 1)
g_sort_by_lcoe = []
g_sort_by_lcoe_rev = []
for i in 1:n_peers_neu
    if g_presorted[i] > 0
        push!(g_sort_by_lcoe, g_presorted[i])
    else
    end
    if g_presorted_rev[i] > 0
        push!(g_sort_by_lcoe_rev, g_presorted_rev[i])
    else
    end
end

sortingUtilityl2h = marginal_utility_buyers_bereinigt_max
presorted_marginal_utility_l2h = sort(sortingUtilityl2h, dims = 2, rev = false)
sortingUtilityh2l = marginal_utility_buyers_bereinigt_max
presorted_marginal_utility_h2l = sort(sortingUtilityh2l, dims = 2, rev = true)
utility_boxplot = []
utility_boxplot_rev = []
for i in 1:n_peers
    if presorted_marginal_utility_l2h[bpt, i] > 0
        push!(utility_boxplot, presorted_marginal_utility_l2h[bpt, i])
    else
    end
    if presorted_marginal_utility_h2l[bpt, i] > 0
        push!(utility_boxplot_rev, presorted_marginal_utility_h2l[bpt, i])
    else
    end
end

utility_d_abgleich = zeros(n_timesteps, n_peers)
for i in 1:n_peers
        for t in 1:n_timesteps
        utility_d_abgleich[t, i] = net_d_results_individual[t, i] * marginal_utility_buyers_bereinigt_max[t, i]
end
end

d_presorted = []
d_presorted_rev = []
for i in 1:n_peers
    for j in 1:n_peers
        if presorted_marginal_utility_l2h[bpt, i] > 0
        if presorted_marginal_utility_l2h[bpt, i] * net_d_results_individual[bpt, j] == utility_d_abgleich[bpt, j]
            push!(d_presorted, net_d_results_individual[bpt, j])
        end
        end
        if presorted_marginal_utility_h2l[bpt, i] > 0
        if presorted_marginal_utility_h2l[bpt, i] * net_d_results_individual[bpt, j] == utility_d_abgleich[bpt, j]
            push!(d_presorted_rev, net_d_results_individual[bpt, j])
        end
        end
    end
end
n_peers_neu2 = size(d_presorted, 1)
d_sort_by_utility = []
d_sort_by_utility_rev = []
for i in 1:n_peers_neu2
    if d_presorted[i] > 0
        push!(d_sort_by_utility, d_presorted[i])
    else
    end
    if d_presorted_rev[i] > 0
        push!(d_sort_by_utility_rev, d_presorted_rev[i])
    else
    end
end




#####SMP Results#####
smp_lambda_res = -shadow_price.(m[:λ])
smp_lambda_res_all = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        smp_lambda_res_all[t, i] = smp_lambda_res[t]
    end
end
smp_gross_totalpayments_sellers_hourly = [sum((g_results_total_hourly[t]) * smp_lambda_res[t] + surplus_revenue_results[t]) for t in 1:n_timesteps]
smp_gross_sellers_total = sum(smp_gross_totalpayments_sellers_hourly[t] for t in 1:n_timesteps)
smp_net_totalpayments_sellers_hourly = [sum(net_g_results_total_hourly[t] * smp_lambda_res[t]) for t in 1:n_timesteps]
smp_net_sellers_total = sum((net_g_results_total_hourly[t]) * smp_lambda_res[t] for t in 1:n_timesteps)
smp_gross_totalpayments_buyers_hourly = [sum((d_results_total_hourly[t] - deficit_results[t]) * smp_lambda_res[t] + deficit_costs_results[t]) for t in 1:n_timesteps]
smp_gross_buyers_total = sum(smp_gross_totalpayments_buyers_hourly[t] for t in 1:n_timesteps)
smp_net_totalpayments_buyers_hourly = [sum(net_d_results_total_hourly[t] * smp_lambda_res[t]) for t in 1:n_timesteps]
smp_net_buyers_total = sum((net_d_results_total_hourly[t]) * smp_lambda_res[t] for t in 1:n_timesteps)



#####PAB Results#####
pab_gross_costs_individual = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_gross_costs_individual[t, i] = lcoe[i] * value.(g)[t,i]
    end
end
pab_net_costs_individual = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_net_costs_individual[t, i] = lcoe[i] * value.(net_g)[t,i]
    end
end
pab_gross_utility_individual = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_gross_utility_individual[t, i] = a[t, i] * value.(d)[t, i]^2 + b[t, i] * value.(d)[t, i] + c[t, i]
    end
end
pab_net_utility_individual = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_net_utility_individual[t, i] = a[t, i] * value.(net_d)[t, i]^2 + b[t, i] * value.(net_d)[t, i] + c[t, i]
    end
end
pab_gross_payments_sellers = pab_gross_costs_individual
pab_gross_payments_buyers = pab_gross_utility_individual
pab_net_payments_sellers = pab_net_costs_individual
pab_net_payments_buyers = pab_net_utility_individual
pab_gross_totalpayments_sellers_hourly = [sum(pab_gross_payments_sellers[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
pab_net_totalpayments_sellers_hourly = [sum(pab_net_payments_sellers[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
pab_gross_totalpayments_buyers_hourly = [sum(pab_gross_payments_buyers[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
pab_net_totalpayments_buyers_hourly = [sum(pab_net_payments_buyers[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
pab_net_revenue_surplus_hourly = pab_net_totalpayments_buyers_hourly - pab_net_totalpayments_sellers_hourly
pab_net_sellers_total = sum(pab_net_payments_sellers[t, i] for t in 1:n_timesteps for i in 1:n_peers)
pab_net_buyers_total = sum(pab_net_payments_buyers[t, i] for t in 1:n_timesteps for i in 1:n_peers)
pab_net_revenue_total = sum(pab_net_revenue_surplus_hourly[t] for t in 1:n_timesteps)
pab_gross_buyers_total = pab_net_buyers_total + deficit_costs_total
pab_gross_sellers_total = pab_net_sellers_total + surplus_revenue_total
pab_buyer_prices = zeros(n_timesteps, n_peers)
pab_seller_prices = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        if net_d_results_individual[t, i] == 0
            pab_buyer_prices[t, i] = 0
        else
            pab_buyer_prices[t, i] = pab_net_payments_buyers[t, i] / net_d_results_individual[t, i]
        end
        if net_g_results_individual[t, i] == 0
            pab_seller_prices[t, i] = 0
        else
            pab_seller_prices[t, i] = pab_net_payments_sellers[t, i] / net_g_results_individual[t, i]
        end
    end
end
pab_average_buyer_price_hourly = [sum(pab_buyer_prices[t, i]/Buyers for i in 1:n_peers) for t in 1:n_timesteps]
pab_average_seller_price_hourly = [sum(pab_seller_prices[t, i] for i in 1:n_peers)/Sellers for t in 1:n_timesteps]



#####PAB Revenue Surplus Compensation Mechanism#####
pab_ΔRev = pab_net_revenue_surplus_hourly
pab_LostRevenueSellers = [sum(net_g_results_individual[t, i] * (smp_lambda_res_all[t, i] - pab_seller_prices[t, i]) for i in 1:n_peers) for t in 1:n_timesteps]
pab_AdditionalCostsBuyers = [sum(net_d_results_individual[t, i] * (pab_buyer_prices[t, i] - smp_lambda_res_all[t, i]) for i in 1:n_peers) for t in 1:n_timesteps]
pab_acb = sum(pab_AdditionalCostsBuyers[t] for t in 1:n_timesteps)
pab_lrs = sum(pab_LostRevenueSellers[t] for t in 1:n_timesteps)

#Mechanism 1: Split over number of peers
pab_TS1a = pab_ΔRev / N
pab_TS1a_boxplot = zeros(n_peers)
for i in 1:n_peers
    pab_TS1a_boxplot[i] = pab_TS1a[bpt]
end
pab_TS1bS = pab_LostRevenueSellers / Sellers
pab_TS1bS_boxplot = zeros(n_peers)
for i in 1:n_peers
    pab_TS1bS_boxplot[i] = pab_TS1bS[bpt]
end
pab_TS1bB = pab_AdditionalCostsBuyers / Buyers
pab_TS1bB_boxplot = zeros(n_peers)
for i in 1:n_peers
    pab_TS1bB_boxplot[i] = pab_TS1bB[bpt]
end
pab_average_TS1a = sum(pab_TS1a[t] for t in 1:n_timesteps)
pab_average_TS1bS = sum(pab_TS1bS[t] for t in 1:n_timesteps)
pab_average_TS1bB = sum(pab_TS1bB[t] for t in 1:n_timesteps)



#Mechanism 2: Split over share of generation and consumption
pab_TS2aS = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_TS2aS[t, i] = (net_g_results_individual[t, i] / Q[t, i]) * pab_ΔRev[t]
    end
end
pab_TS2aS_boxplot = pab_TS2aS[bpt, :]
pab_TS2aB = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_TS2aB[t, i] = (net_d_results_individual[t, i] / Q[t, i]) * pab_ΔRev[t]
    end
end
pab_TS2aB_boxplot = []
for i in 1:n_peers
    if pab_TS2aB[bpt, i] >= 0.0000001
        push!(pab_TS2aB_boxplot, pab_TS2aB[bpt, i])
    end
end
pab_TS2bS = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_TS2bS[t, i] = net_g_results_individual[t, i] / net_g_results_total_hourly[t] * pab_LostRevenueSellers[t]
    end
end
pab_TS2bS_boxplot = []
for i in 1:n_peers
    if pab_TS2bS[bpt, i] >= 0.0000001
        push!(pab_TS2bS_boxplot, pab_TS2bS[bpt, i])
    end
end
pab_TS2bB = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        pab_TS2bB[t, i] = net_d_results_individual[t, i] / net_d_results_total_hourly[t] * pab_AdditionalCostsBuyers[t]
    end
end
pab_TS2bB_boxplot = []
for i in 1:n_peers
    if pab_TS2bB[bpt, i] >= 0.0000001
        push!(pab_TS2bB_boxplot, pab_TS2bB[bpt, i])
    end
end

pab_average_TS2aS = sum(pab_TS2aS[t, i] for t in 1:n_timesteps for i in 1:n_peers)
pab_average_TS2aB = sum(pab_TS2aB[t, i] for t in 1:n_timesteps for i in 1:n_peers)
pab_average_TS2bS = sum(pab_TS2bS[t, i] for t in 1:n_timesteps for i in 1:n_peers)
pab_average_TS2bB = sum(pab_TS2bB[t, i] for t in 1:n_timesteps for i in 1:n_peers)

#Mechanism 3: Split over share of cost and utility
length_neu = size(lcoe_boxplot, 1)
pab_TS3aIS = zeros(length_neu)
for i in 1:length_neu
    pab_TS3aIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_unbereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * pab_ΔRev[bpt]
end
pab_TS3aIIS = zeros(length_neu)
for i in 1:length_neu
    pab_TS3aIIS[i] = ((lcoe_boxplot_rev[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot_rev[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_unbereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * pab_ΔRev[bpt]
end
pab_TS3aIIIS = zeros(length_neu)
for i in 1:length_neu
    pab_TS3aIIIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_unbereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * pab_ΔRev[bpt]
end
pab_TS3aIS_boxplot = []
for i in 1:length_neu
        push!(pab_TS3aIS_boxplot, pab_TS3aIS[i])
end
pab_TS3aIIS_boxplot = []
for i in 1:length_neu
        push!(pab_TS3aIIS_boxplot, pab_TS3aIIS[i])
end
pab_TS3aIIIS_boxplot = []
for i in 1:length_neu
        push!(pab_TS3aIIIS_boxplot, pab_TS3aIIIS[i])
end

pab_TS3aIB = zeros(n_peers)
for i in 1:n_peers
    pab_TS3aIB[i] = ((marginal_utility_buyers_bereinigt[bpt, i] * net_d_results_individual[bpt, i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_bereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * pab_ΔRev[bpt]
end
pab_TS3aIIB = zeros(n_peers)
for i in 1:n_peers
    pab_TS3aIIB[i] = ((marginal_utility_buyers_bereinigt[bpt, i] * net_d_results_individual[bpt, i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_bereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * pab_ΔRev[bpt]
end
pab_TS3aIIIB = zeros(n_peers)
for i in 1:n_peers
    pab_TS3aIIIB[i] = ((marginal_utility_buyers_bereinigt[bpt, i] * net_d_results_individual[bpt, i]) / (sum(lcoe_boxplot_rev[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_bereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * pab_ΔRev[bpt]
end
pab_TS3aIB_boxplot = []
for i in 1:n_peers
    if pab_TS3aIB[i] > 0.0000001
        push!(pab_TS3aIB_boxplot, pab_TS3aIB[i])
    end
end
pab_TS3aIIB_boxplot = []
for i in 1:n_peers
    if pab_TS3aIB[i] > 0.0000001
        push!(pab_TS3aIIB_boxplot, pab_TS3aIIB[i])
    end
end
pab_TS3aIIIB_boxplot = []
for i in 1:n_peers
    if pab_TS3aIB[i] > 0.0000001
        push!(pab_TS3aIIIB_boxplot, pab_TS3aIIIB[i])
    end
end


#CDS:
pab_TS3bIS = zeros(length_neu)
for i in 1:length_neu
    pab_TS3bIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu))) * pab_LostRevenueSellers[bpt]
end
pab_TS3bIIS = zeros(length_neu)
for i in 1:length_neu
    pab_TS3bIIS[i] = ((lcoe_boxplot_rev[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot_rev[i] * g_sort_by_lcoe[i] for i in 1:length_neu))) * pab_LostRevenueSellers[bpt]
end
pab_TS3bIIIS = zeros(length_neu)
for i in 1:length_neu
    pab_TS3bIIIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu))) * pab_LostRevenueSellers[bpt]
end

pab_TS3bIS_boxplot = []
for i in 1:length_neu
        push!(pab_TS3bIS_boxplot, pab_TS3bIS[i])
end
pab_TS3bIIS_boxplot = []
for i in 1:length_neu
        push!(pab_TS3bIIS_boxplot, pab_TS3bIIS[i])
end
pab_TS3bIIIS_boxplot = []
for i in 1:length_neu
        push!(pab_TS3bIIIS_boxplot, pab_TS3bIIIS[i])
end


length_neu2 = size(utility_boxplot, 1)
pab_TS3bIB = zeros(length_neu2)
for i in 1:length_neu2
    pab_TS3bIB[i] = ((utility_boxplot[i] * d_sort_by_utility[i]) / (sum(utility_boxplot[j] * d_sort_by_utility[j] for j in 1:length_neu2))) * pab_AdditionalCostsBuyers[bpt]
end
pab_TS3bIIB = zeros(length_neu2)
for i in 1:length_neu2
    pab_TS3bIIB[i] = ((utility_boxplot[i] * d_sort_by_utility[i]) / (sum(utility_boxplot[j] * d_sort_by_utility[j] for j in 1:length_neu2))) * pab_AdditionalCostsBuyers[bpt]
end
pab_TS3bIIIB = zeros(length_neu2)
for i in 1:length_neu2
    pab_TS3bIIIB[i] = ((utility_boxplot_rev[i] * d_sort_by_utility[i]) / (sum(utility_boxplot_rev[j] * d_sort_by_utility[j] for j in 1:length_neu2))) * pab_AdditionalCostsBuyers[bpt]
end
pab_TS3bIB_boxplot = []
for i in 1:length_neu2
    if pab_TS3bIB[i] > 0
        push!(pab_TS3bIB_boxplot, pab_TS3bIB[i])
    end
end
pab_TS3bIIB_boxplot = []
for i in 1:length_neu2
    if pab_TS3bIIB[i] > 0
        push!(pab_TS3bIIB_boxplot, pab_TS3bIIB[i])
    end
end
pab_TS3bIIIB_boxplot = []
for i in 1:length_neu2
    if pab_TS3bIIIB[i] > 0
        push!(pab_TS3bIIIB_boxplot, pab_TS3bIIIB[i])
    end
end

#####VCG Optimization: Computation of Prices#####
vcg_highest_costs_settled = zeros(n_timesteps)
for t in 1:n_timesteps, i in 1:n_peers
    if g_results_individual[t, i] == 0
        vcg_highest_costs_settled[t] = 0.3 / 1000
    else
        vcg_highest_costs_settled[t] = maximum(marginal_price_sellers_bereinigt[t, :])
    end
end

vcg_gmax = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        if lcoe[i] * g_results_individual[t, i] == vcg_highest_costs_settled[t] * g_results_individual[t, i]
            vcg_gmax[t, i] = 0
        else
            vcg_gmax[t, i] = peers[i].gmax[t]
        end
    end
end

vcg_dmax = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        vcg_dmax[t, i] = peers[i].dmax[t]
    end
end

vcg_max_gen_hourly = [sum(vcg_gmax[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
vcg_max_dem_hourly = [sum(vcg_dmax[t, i] for i in 1:n_peers) for t in 1:n_timesteps]

vcg = Model(with_optimizer(Gurobi.Optimizer))

@variable(vcg, vcg_g[1:n_timesteps, 1:n_peers]) #generation
@variable(vcg, vcg_d[1:n_timesteps, 1:n_peers]) #demand
@variable(vcg, vcg_deficit[1:n_timesteps]) #Consumption from Grid
@variable(vcg, vcg_surplus[1:n_timesteps]) #Excess generation feed into grid

for t in 1:n_timesteps
    if vcg_max_gen_hourly[t] >= vcg_max_dem_hourly[t]
        @constraint(vcg, vcg_deficit[t] == 0)
    else
        @constraint(vcg, vcg_surplus[t] == 0)
    end
end

@constraint(vcg, [t = 1:n_timesteps], sum(vcg_d[t, i] for i in 1:n_peers) + vcg_surplus[t] == sum(vcg_g[t, i] for i in 1:n_peers) + vcg_deficit[t])

@constraint(vcg, [t = 1:n_timesteps, i = 1:n_peers], vcg_g[t, i] <= vcg_gmax[t, i])
@constraint(vcg, [t = 1:n_timesteps, i = 1:n_peers], vcg_g[t, i] >= peers[i].gmin[t])

@constraint(vcg, [t = 1:n_timesteps, i = 1:n_peers], vcg_d[t, i] <= vcg_dmax[t, i])
@constraint(vcg, [t = 1:n_timesteps, i = 1:n_peers], vcg_d[t, i] >= peers[i].dmin[t])

@constraint(vcg, [t = 1:n_timesteps], vcg_surplus[t] >= 0)
@constraint(vcg, [t = 1:n_timesteps], vcg_deficit[t] >= 0)

vcg_generation_costs_total = [sum(lcoe[i] * vcg_g[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
vcg_demand_utility_total = [sum(a[t, i] * vcg_d[t, i]^2 + b[t, i] * vcg_d[t, i] + c[t, i] for i in 1:n_peers) for t in 1:n_timesteps]

@expression(vcg, vcg_deficit_costs[t = 1:n_timesteps], grid_tariff * vcg_deficit[t])
@expression(vcg, vcg_surplus_revenue[t = 1:n_timesteps], feedin_tariff * vcg_surplus[t])

@objective(vcg, Min, sum(vcg_generation_costs_total[t] + vcg_deficit_costs[t] - vcg_demand_utility_total[t] - vcg_surplus_revenue[t] for t in 1:n_timesteps))

optimize!(vcg)
status = termination_status(vcg)

obv = objective_value(vcg)



#####VCG Results#####
vcg_d_results_individual = value.(vcg_d)
vcg_d_results_total = [sum(vcg_d_results_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
vcg_g_results_individual = value.(vcg_g)
vcg_g_results_total = [sum(vcg_g_results_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
vcg_deficit_results = value.(vcg_deficit)
vcg_surplus_results = value.(vcg_surplus)
vcg_generation_costs_individual_results = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        vcg_generation_costs_individual_results[t, i] = lcoe[i] * vcg_g_results_individual[t, i]
    end
end
vcg_demand_utility_individual_results = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        vcg_demand_utility_individual_results[t, i] = a[t, i] * vcg_d_results_individual[t, i]^2 + b[t, i] * vcg_d_results_individual[t, i] + c[t, i]
    end
end
vcg_marginal_price_sellers_bereinigt = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        if vcg_g_results_individual[t, i] > 0
        vcg_marginal_price_sellers_bereinigt[t, i] = lcoe_all[t, i]
        else
        vcg_marginal_price_sellers_bereinigt[t, i] = 0
        end
    end
end

vcg_lowest_costs_unsettled = zeros(n_timesteps)
for t in 1:n_timesteps, i in 1:n_peers
    if maximum(vcg_marginal_price_sellers_bereinigt[t, :]) <= maximum(marginal_price_sellers_bereinigt[t, :])
        vcg_lowest_costs_unsettled[t] = 0.3 / 1000
    else
        vcg_lowest_costs_unsettled[t] = maximum(vcg_marginal_price_sellers_bereinigt[t, :])
    end
end

lcoe_2nd = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps, i in 1:n_peers
    if g_results_individual[t, i] > 0
        if vcg_highest_costs_settled[t] == marginal_price_sellers_bereinigt[t, i]
            lcoe_2nd[t, i] = 0
        else
            lcoe_2nd[t, i] = marginal_price_sellers_bereinigt[t, i]
        end
    end
end
vcg_lowest_utility_settled = vcg_lowest_costs_unsettled
vcg_highest_utility_unsettled = zeros(n_timesteps)
for t in 1:n_timesteps, i in 1:n_peers
    if smp_lambda_res[t] > 0.29 / 1000
        vcg_highest_utility_unsettled[t] = 0.3 / 1000
    else
        vcg_highest_utility_unsettled[t] = maximum(lcoe_2nd[t, :])
    end
end

vcg_buyer_price = zeros(n_timesteps)
for t in 1:n_timesteps
    for i in 1:n_peers
        if smp_lambda_res[t] >= 0.3 / 1000
            vcg_buyer_price[t] == smp_lambda_res[t]
        else
            vcg_buyer_price[t] = max(vcg_highest_utility_unsettled[t], vcg_highest_costs_settled[t])
        end
    end
end

vcg_seller_price = zeros(n_timesteps)
for t in 1:n_timesteps
    for i in 1:n_peers
        if smp_lambda_res[t] >= 0.3 / 1000
            vcg_seller_price[t] == smp_lambda_res[t]
        else
            vcg_seller_price[t] = min(vcg_lowest_costs_unsettled[t], vcg_lowest_utility_settled[t])
        end
    end
end

vcg_buyer_payment_individual = vcg_buyer_price .* net_d_results_individual
vcg_seller_payment_individual = vcg_seller_price .* net_g_results_individual
vcg_buyers_hourly = [sum(vcg_buyer_payment_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
vcg_sellers_hourly = [sum(vcg_seller_payment_individual[t, i] for i in 1:n_peers) for t in 1:n_timesteps]
vcg_buyers_total = sum(vcg_buyer_payment_individual[t, i] for t in 1:n_timesteps for i in 1:n_peers)
vcg_sellers_total = sum(vcg_seller_payment_individual[t, i] for t in 1:n_timesteps for i in 1:n_peers)
vcg_revenue_hourly = vcg_buyers_hourly - vcg_sellers_hourly
vcg_revenue_total = vcg_buyers_total - vcg_sellers_total
vcg_gross_buyers_total = vcg_buyers_total + deficit_costs_total
vcg_gross_sellers_total = vcg_sellers_total + surplus_revenue_total



#####VCG Revenue Deficit Compensation Mechanism######
vcg_ΔRev = -vcg_revenue_hourly
vcg_AdditionalRevenueSellers = [sum(net_g_results_individual[t, i] * (vcg_seller_price[t] - smp_lambda_res[t]) for i in 1:n_peers) for t in 1:n_timesteps]
vcg_AdditionalCostBuyers = [sum(net_d_results_individual[t, i] * (smp_lambda_res[t] - vcg_buyer_price[t]) for i in 1:n_peers) for t in 1:n_timesteps]
vcg_lcb = sum(vcg_AdditionalCostBuyers[t] for t in 1:n_timesteps)
vcg_ars = sum(vcg_AdditionalRevenueSellers[t] for t in 1:n_timesteps)

#Mechanism 1: Split over number of peers
vcg_TF1a = vcg_ΔRev / N
vcg_TF1a_boxplot = zeros(n_peers)
for i in 1:n_peers
    vcg_TF1a_boxplot[i] = vcg_TF1a[bpt]
end
vcg_TF1bS = vcg_AdditionalRevenueSellers / Sellers
vcg_TF1bS_boxplot = zeros(n_peers)
for i in 1:n_peers
    vcg_TF1bS_boxplot[i] = vcg_TF1bS[bpt]
end
vcg_TF1bB = vcg_AdditionalCostBuyers / Buyers
vcg_TF1bB_boxplot = zeros(n_peers)
for i in 1:n_peers
    vcg_TF1bB_boxplot[i] = vcg_TF1bB[bpt]
end

#Mechanism 2: Split over share of generation and consumption
vcg_TF2aS = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        vcg_TF2aS[t, i] = (net_g_results_individual[t, i] / Q[t, i]) * vcg_ΔRev[t]
    end
end
vcg_TF2aS_boxplot = []
for i in 1:n_peers
    if vcg_TF2aS[bpt, i] >= 0.0000001
        push!(vcg_TF2aS_boxplot, vcg_TF2aS[bpt, i])
    end
end
vcg_TF2aB = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        vcg_TF2aB[t, i] = (net_d_results_individual[t, i] / Q[t, i]) * vcg_ΔRev[t]
    end
end
vcg_TF2aB_boxplot = []
for i in 1:n_peers
    if vcg_TF2aB[bpt, i] >= 0.0000001
        push!(vcg_TF2aB_boxplot, vcg_TF2aB[bpt, i])
    end
end

vcg_TF2bS = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        vcg_TF2bS[t, i] = net_g_results_individual[t, i] / net_g_results_total_hourly[t] * vcg_AdditionalRevenueSellers[t]
    end
end
vcg_TF2bS_boxplot = []
for i in 1:n_peers
    if vcg_TF2bS[bpt, i] >= 0.0000001
        push!(vcg_TF2bS_boxplot, vcg_TF2bS[bpt, i])
    end
end
vcg_TF2bB = zeros(n_timesteps, n_peers)
for t in 1:n_timesteps
    for i in 1:n_peers
        vcg_TF2bB[t, i] = net_d_results_individual[t, i] / net_d_results_total_hourly[t] * vcg_AdditionalCostBuyers[t]
    end
end
vcg_TF2bB_boxplot = []
for i in 1:n_peers
    if vcg_TF2bB[bpt, i] >= 0.0000001
        push!(vcg_TF2bB_boxplot, vcg_TF2bB[bpt, i])
    end
end

#Mechanism 3: Split over share of cost and utility
#EDS:
length_neu = size(lcoe_boxplot, 1)
vcg_TF3aIS = zeros(length_neu)
for i in 1:length_neu
    vcg_TF3aIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_unbereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * vcg_ΔRev[bpt]
end
vcg_TF3aIIS = zeros(length_neu)
for i in 1:length_neu
    vcg_TF3aIIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_unbereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * vcg_ΔRev[bpt]
end
vcg_TF3aIIIS = zeros(length_neu)
for i in 1:length_neu
    vcg_TF3aIIIS[i] = ((lcoe_boxplot_rev[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot_rev[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_unbereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * vcg_ΔRev[bpt]
end
vcg_TF3aIS_boxplot = []
for i in 1:length_neu
        push!(vcg_TF3aIS_boxplot, vcg_TF3aIS[i])
end
vcg_TF3aIIS_boxplot = []
for i in 1:length_neu
        push!(vcg_TF3aIIS_boxplot, vcg_TF3aIIS[i])
end
vcg_TF3aIIIS_boxplot = []
for i in 1:length_neu
        push!(vcg_TF3aIIIS_boxplot, vcg_TF3aIIIS[i])
end

vcg_TF3aIB = zeros(n_peers)
for i in 1:n_peers
    vcg_TF3aIB[i] = ((marginal_utility_buyers_bereinigt[bpt, i] * net_d_results_individual[bpt, i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_bereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * vcg_ΔRev[bpt]
end
vcg_TF3aIIB = zeros(n_peers)
for i in 1:n_peers
    vcg_TF3aIIB[i] = ((marginal_utility_buyers_bereinigt[bpt, i] * net_d_results_individual[bpt, i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_bereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * vcg_ΔRev[bpt]
end
vcg_TF3aIIIB = zeros(n_peers)
for i in 1:n_peers
    vcg_TF3aIIIB[i] = ((marginal_utility_buyers_bereinigt[bpt, i] * net_d_results_individual[bpt, i]) / (sum(lcoe_boxplot_rev[i] * g_sort_by_lcoe[i] for i in 1:length_neu) + sum(marginal_utility_buyers_bereinigt[bpt, j] * net_d_results_individual[bpt, j] for j in 1:n_peers))) * vcg_ΔRev[bpt]
end
vcg_TF3aIB_boxplot = []
for i in 1:n_peers
    if vcg_TF3aIB[i] > 0.0000001
        push!(vcg_TF3aIB_boxplot, vcg_TF3aIB[i])
    end
end
vcg_TF3aIIB_boxplot = []
for i in 1:n_peers
    if vcg_TF3aIB[i] > 0.0000001
        push!(vcg_TF3aIIB_boxplot, vcg_TF3aIIB[i])
    end
end
vcg_TF3aIIIB_boxplot = []
for i in 1:n_peers
    if vcg_TF3aIB[i] > 0.0000001
        push!(vcg_TF3aIIIB_boxplot, vcg_TF3aIIIB[i])
    end
end

#CDS:
vcg_TF3bIS = zeros(length_neu)
for i in 1:length_neu
    vcg_TF3bIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu))) * vcg_AdditionalRevenueSellers[bpt]
end
vcg_TF3bIIS = zeros(length_neu)
for i in 1:length_neu
    vcg_TF3bIIS[i] = ((lcoe_boxplot[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot[i] * g_sort_by_lcoe[i] for i in 1:length_neu))) * vcg_AdditionalRevenueSellers[bpt]
end
vcg_TF3bIIIS = zeros(length_neu)
for i in 1:length_neu
    vcg_TF3bIIIS[i] = ((lcoe_boxplot_rev[i] * g_sort_by_lcoe[i]) / (sum(lcoe_boxplot_rev[i] * g_sort_by_lcoe[i] for i in 1:length_neu))) * vcg_AdditionalRevenueSellers[bpt]
end

vcg_TF3bIS_boxplot = []
for i in 1:length_neu
        push!(vcg_TF3bIS_boxplot, vcg_TF3bIS[i])
end
vcg_TF3bIIS_boxplot = []
for i in 1:length_neu
        push!(vcg_TF3bIIS_boxplot, vcg_TF3bIIS[i])
end
vcg_TF3bIIIS_boxplot = []
for i in 1:length_neu
        push!(vcg_TF3bIIIS_boxplot, vcg_TF3bIIIS[i])
end

length_neu2 = size(utility_boxplot, 1)
vcg_TF3bIB = zeros(length_neu2)
for i in 1:length_neu2
    vcg_TF3bIB[i] = ((utility_boxplot[i] * d_sort_by_utility[i]) / (sum(utility_boxplot[j] * d_sort_by_utility[j] for j in 1:length_neu2))) * vcg_AdditionalCostBuyers[bpt]
end
vcg_TF3bIIB = zeros(length_neu2)
for i in 1:length_neu2
    vcg_TF3bIIB[i] = ((utility_boxplot_rev[i] * d_sort_by_utility[i]) / (sum(utility_boxplot_rev[j] * d_sort_by_utility[j] for j in 1:length_neu2))) * vcg_AdditionalCostBuyers[bpt]
end
vcg_TF3bIIIB = zeros(length_neu2)
for i in 1:length_neu2
    vcg_TF3bIIIB[i] = ((utility_boxplot[i] * d_sort_by_utility[i]) / (sum(utility_boxplot[j] * d_sort_by_utility[j] for j in 1:length_neu2))) * vcg_AdditionalCostBuyers[bpt]
end
vcg_TF3bIB_boxplot = []
for i in 1:length_neu2
    if vcg_TF3bIB[i] > 0
        push!(vcg_TF3bIB_boxplot, vcg_TF3bIB[i])
    end
end
vcg_TF3bIIB_boxplot = []
for i in 1:length_neu2
    if vcg_TF3bIIB[i] > 0
        push!(vcg_TF3bIIB_boxplot, vcg_TF3bIIB[i])
    end
end
vcg_TF3bIIIB_boxplot = []
for i in 1:length_neu2
    if vcg_TF3bIIIB[i] > 0
        push!(vcg_TF3bIIIB_boxplot, vcg_TF3bIIIB[i])
    end
end



#####Data Analysis#####
# pyplot(size = (300,300), legend = true)
# font = Plots.font("Times New Roman", 6)
# font_compensationplots = Plots.font("Times New Roman", 4)
# font_legend = Plots.font("Times New Roman", 3)
# pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font_legend)

#loadprofile=plot([d_results_total_hourly/1000, max_gen_hourly/1000, NetGenSurplus/1000], label=["Demand" "Generation" "NetGenerationSurplus"],lw=0.5, xlabel="Timestep", ylabel="kWh", foreground_color_grid=:lightgrey, legend=:topright)
#seller_revenue_distribution = pie(sizes,labels=labels,shadow=true,startangle=90,explode=explode,colors=colors,autopct="%1.1f%%",textprops=font)
#buyer_cost_distribution = pie([],labels=labels,shadow=true,startangle=90,explode=explode,colors=colors,autopct="%1.1f%%",textprops=font)
# smp_price_plot = plot([smp_lambda_res*1000], label="Ø-Buyer/Seller-Price",lw=0.5, xlabel="Timestep", ylabel="\$/kWh", foreground_color_grid=:lightgrey)
# pab_price_plot = plot([pab_average_buyer_price_hourly*1000, pab_average_seller_price_hourly*1000], label=["Ø-Buyer-Price" "Ø-Seller-Price"],lw=0.5, xlabel="Timestep", ylabel="\$/kWh", foreground_color_grid=:lightgrey)
# vcg_price_plot = plot([vcg_buyer_price*1000, vcg_seller_price*1000], label=["Ø-Buyer-Price" "Ø-Seller-Price"],lw=0.5, xlabel="Timestep", ylabel="\$/kWh", foreground_color_grid=:lightgrey)
# vcg_edm_seller = boxplot([vcg_TF1a_boxplot, vcg_TF2aS_boxplot, vcg_TF3aIS_boxplot, vcg_TF3aIIS_boxplot, vcg_TF3aIIIS_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)
# vcg_edm_buyer = boxplot([vcg_TF1a_boxplot, vcg_TF2aB_boxplot, vcg_TF3aIB_boxplot, vcg_TF3aIIIB_boxplot, vcg_TF3aIIB_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)
# vcg_cpm_seller = boxplot([vcg_TF1bS_boxplot, vcg_TF2bS_boxplot, vcg_TF3bIS_boxplot, vcg_TF3bIIS_boxplot, vcg_TF3bIIIS_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)
# vcg_cpm_buyer = boxplot([vcg_TF1bB_boxplot, vcg_TF2bB_boxplot, vcg_TF3bIB_boxplot, vcg_TF3bIIB_boxplot, vcg_TF3bIIIB_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)
# pab_edm_seller = boxplot([pab_TS1a_boxplot, pab_TS2aS_boxplot, pab_TS3aIS_boxplot, pab_TS3aIIS_boxplot, pab_TS3aIIIS_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)
# pab_edm_buyer = boxplot([pab_TS1a_boxplot, pab_TS2aB_boxplot, pab_TS3aIB_boxplot, pab_TS3aIIB_boxplot, pab_TS3aIIIB_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)
# pab_cpm_seller = boxplot([pab_TS1bS_boxplot, pab_TS2bS_boxplot, pab_TS3bIS_boxplot, pab_TS3bIIS_boxplot, pab_TS3bIIIS_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)
# pab_cpm_buyer = boxplot([pab_TS1bB_boxplot, pab_TS2bB_boxplot, pab_TS3bIB_boxplot, pab_TS3bIIB_boxplot, pab_TS3bIIIB_boxplot], leg=false, xticks = ([1:1:5;], ["S-AM", "A-AM", "PP-AM", "WC-AM","SF-AM"]), thickness_scaling = 1, foreground_color_grid=:lightgrey)



# savefig(loadprofile, "Loadprofile.eps")
# savefig(smp_price_plot, "Average_smp.eps")
# savefig(pab_price_plot, "Average_pab.eps")
# savefig(vcg_price_plot, "Average_vcg.eps")
# savefig(vcg_edm_seller, "vcg_edm_seller.eps")
# savefig(vcg_edm_buyer, "vcg_edm_buyer.eps")
# savefig(vcg_cpm_seller, "vcg_cpm_seller.eps")
# savefig(vcg_cpm_buyer, "vcg_cpm_buyer.eps")
# savefig(pab_edm_seller, "pab_edm_seller.eps")
# savefig(pab_edm_buyer, "pab_edm_buyer.eps")
# savefig(pab_cpm_seller, "pab_cpm_seller.eps")
# savefig(pab_cpm_buyer, "pab_cpm_buyer.eps")


mutable struct Peer
    index::Int
    type::Any
    gmax::Vector{Float64}
    gmin::Vector{Float64}
    dmax::Vector{Float64}
    dmin::Vector{Float64}
    test::Vector{Float64}
    c0::Float64
    c1::Float64
    c2::Float64
    function Peer(index, type, gmax_series, gmin_series, dmax_series, dmin_series, test_series; c0=0, c1=0, c2=0)
        p = new()
        p.index = index
        p.type = type
        @assert length(dmax_series) == length(dmin_series)
        p.dmax = dmax_series
        p.dmin = dmin_series
        @assert length(gmax_series) == length(gmin_series)
        p.gmax = gmax_series
        p.gmin = gmin_series
        @assert length(gmax_series) == length(test_series)
        p.c0 = c0
        p.c1 = c1
        p.c2 = c2
        p.test = test_series
        return p
    end
end

mutable struct DataCollection
    timesteps::Vector{Int}
    n_timesteps::Int
    peers::Vector{Peer}
    n_peers::Int
    function DataCollection(timesteps, peers)
        dc = new()
        dc.timesteps = timesteps
        dc.n_timesteps = length(timesteps)
        dc.peers = peers
        dc.n_peers = length(peers)
        return dc
    end
end


function load_data(input_file)
    data_raw = DataFrame(CSV.read(input_file))
    dropmissing!(data_raw)

    time_steps = unique(data_raw[:Hour])
    peer_ids = unique(data_raw[Symbol("Buyer/Seller")])

    peers  = []

    for p_id in peer_ids
        pdata = data_raw[data_raw[Symbol("Buyer/Seller")] .== p_id, :]
        gmin = pdata[Symbol("Gmin [Wh]")]
        gmax = pdata[Symbol("Gmax [Wh]")]
        dmin = pdata[Symbol("Dmin [Wh]")]
        dmax = pdata[Symbol("Dmax [Wh]")]
        test = pdata[Symbol("v")]
        p = Peer(p_id, "peer",  gmax, gmin, dmax, dmin, test)
        push!(peers, p)
    end

    return DataCollection(time_steps, peers)
end

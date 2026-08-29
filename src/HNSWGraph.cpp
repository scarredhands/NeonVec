#include "HNSWGraph.h"
#include <algorithm>
#include <cmath>
#include <iostream>

HNSWGraph::HNSWGraph(VectorStorage& db, size_t max_edges, size_t ef_cons) 
    : storage(db), M(max_edges), M_max(max_edges), M_max0(max_edges * 2), 
      ef_construction(ef_cons), max_layer(-1), entry_point(0) {
    
    level_mult = 1.0 / log(1.0 * M);
    rng.seed(1337); 
}

int HNSWGraph::get_random_level() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(rng)) * level_mult;
    return (int)r;
}

std::vector<std::pair<float, size_t>> HNSWGraph::search_layer(
    const std::vector<int8_t>& query, size_t ep, size_t ef, int layer) const {
    
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<>> candidates;
    std::priority_queue<std::pair<float, size_t>> top_results;
    std::unordered_set<size_t> visited;

    const int8_t* q_ptr = query.data();
    float entry_dist = storage.compute_l2_sq(q_ptr, storage.get_vector(ep), storage.get_dim());
    
    candidates.push({entry_dist, ep});
    
    // Only push entry point to results if it's alive
    if (!storage.is_deleted(ep)) {
        top_results.push({entry_dist, ep});
    }
    
    visited.insert(ep);

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        if (!top_results.empty() && current.first > top_results.top().first && top_results.size() == ef) break;

        for (size_t neighbor_id : graph[layer][current.second]) {
            if (visited.find(neighbor_id) == visited.end()) {
                visited.insert(neighbor_id);
                float dist = storage.compute_l2_sq(q_ptr, storage.get_vector(neighbor_id), storage.get_dim());
                
                // 1. ALWAYS push to candidates (Drive through the ghost town)
                candidates.push({dist, neighbor_id});
                
                // 2. ONLY push to top_results if the node is alive (Don't stop in the ghost town)
                if (!storage.is_deleted(neighbor_id)) {
                    if (top_results.size() < ef || dist < top_results.top().first) {
                        top_results.push({dist, neighbor_id});
                        if (top_results.size() > ef) top_results.pop();
                    }
                }
            }
        }
    }

    std::vector<std::pair<float, size_t>> results;
    while (!top_results.empty()) {
        results.push_back(top_results.top());
        top_results.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

void HNSWGraph::insert(size_t new_vid) {
    int new_level = get_random_level();
    
    const int8_t* raw_vec = storage.get_vector(new_vid);
    std::vector<int8_t> query(raw_vec, raw_vec + storage.get_dim());
    
    if (new_vid >= node_levels.size()) {
        size_t new_size = new_vid + 1000;
        node_levels.resize(new_size, -1);
        for (auto& layer : graph) {
            layer.resize(new_size);
        }
    }
    node_levels[new_vid] = new_level;

    while (graph.size() <= (size_t)new_level) {
        graph.push_back(std::vector<std::vector<size_t>>(node_levels.size()));
    }

    if (new_vid == 0) {
        max_layer = new_level;
        entry_point = 0;
        return;
    }

    size_t curr_ep = entry_point;
    
    for (int lc = max_layer; lc > new_level; --lc) {
        auto nearest = search_layer(query, curr_ep, 1, lc); 
        if (!nearest.empty()) curr_ep = nearest[0].second;
    }

    for (int lc = std::min(max_layer, new_level); lc >= 0; --lc) {
        auto nearest = search_layer(query, curr_ep, ef_construction, lc);
        if (nearest.empty()) continue;
        
        curr_ep = nearest[0].second; 
        size_t max_conn = (lc == 0) ? M_max0 : M_max;
        size_t connections = std::min(M, nearest.size());

        for (size_t i = 0; i < connections; ++i) {
            size_t neighbor_id = nearest[i].second;
            
            graph[lc][new_vid].push_back(neighbor_id);
            graph[lc][neighbor_id].push_back(new_vid);
            
            if (graph[lc][neighbor_id].size() > max_conn) {
                graph[lc][neighbor_id].erase(graph[lc][neighbor_id].begin());
            }
        }
    }

    if (new_level > max_layer) {
        max_layer = new_level;
        entry_point = new_vid;
    }
}

std::vector<std::pair<float, size_t>> HNSWGraph::search_ann(
    const std::vector<float>& query_float, size_t k, size_t ef_search) const {
    
    if (max_layer == -1) return {};
    std::vector<int8_t> query = VectorStorage::quantize(query_float);
    size_t curr_ep = entry_point;

    for (int lc = max_layer; lc > 0; --lc) {
        auto nearest = search_layer(query, curr_ep, 1, lc);
        if (!nearest.empty()) curr_ep = nearest[0].second;
    }

    return search_layer(query, curr_ep, std::max(k, ef_search), 0);
}

// ==========================================
// BACKGROUND GARBAGE COLLECTION THREAD
// ==========================================
void HNSWGraph::prune_tombstones() {
    // Iterate through every layer in the graph
    for (int lc = 0; lc <= max_layer; ++lc) {
        // Iterate through every node in this layer
        for (size_t node_id = 0; node_id < graph[lc].size(); ++node_id) {
            
            auto& neighbors = graph[lc][node_id];
            
            // C++ Erase-Remove Idiom: Fast, in-place filtering of deleted neighbors
            neighbors.erase(
                std::remove_if(neighbors.begin(), neighbors.end(),
                    [this](size_t id) { return storage.is_deleted(id); }
                ),
                neighbors.end()
            );
        }
    }
}

// ==========================================
// SNAPSHOT MEMORY DUMP LOGIC
// ==========================================
void HNSWGraph::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&max_layer), sizeof(int));
    out.write(reinterpret_cast<const char*>(&entry_point), sizeof(size_t));
    
    size_t nl_size = node_levels.size();
    out.write(reinterpret_cast<const char*>(&nl_size), sizeof(size_t));
    if (nl_size > 0) out.write(reinterpret_cast<const char*>(node_levels.data()), nl_size * sizeof(int));
    
    size_t g_layers = graph.size();
    out.write(reinterpret_cast<const char*>(&g_layers), sizeof(size_t));
    for (size_t l = 0; l < g_layers; ++l) {
        size_t g_nodes = graph[l].size();
        out.write(reinterpret_cast<const char*>(&g_nodes), sizeof(size_t));
        for (size_t n = 0; n < g_nodes; ++n) {
            size_t e_size = graph[l][n].size();
            out.write(reinterpret_cast<const char*>(&e_size), sizeof(size_t));
            if (e_size > 0) out.write(reinterpret_cast<const char*>(graph[l][n].data()), e_size * sizeof(size_t));
        }
    }
}

void HNSWGraph::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&max_layer), sizeof(int));
    in.read(reinterpret_cast<char*>(&entry_point), sizeof(size_t));
    
    size_t nl_size;
    in.read(reinterpret_cast<char*>(&nl_size), sizeof(size_t));
    node_levels.resize(nl_size);
    if (nl_size > 0) in.read(reinterpret_cast<char*>(node_levels.data()), nl_size * sizeof(int));
    
    size_t g_layers;
    in.read(reinterpret_cast<char*>(&g_layers), sizeof(size_t));
    graph.resize(g_layers);
    for (size_t l = 0; l < g_layers; ++l) {
        size_t g_nodes;
        in.read(reinterpret_cast<char*>(&g_nodes), sizeof(size_t));
        graph[l].resize(g_nodes);
        for (size_t n = 0; n < g_nodes; ++n) {
            size_t e_size;
            in.read(reinterpret_cast<char*>(&e_size), sizeof(size_t));
            graph[l][n].resize(e_size);
            if (e_size > 0) in.read(reinterpret_cast<char*>(graph[l][n].data()), e_size * sizeof(size_t));
        }
    }
}
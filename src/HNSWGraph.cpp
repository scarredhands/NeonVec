#include "HNSWGraph.h"
#include <algorithm>
#include <cmath>
#include <iostream>

HNSWGraph::HNSWGraph(VectorStorage& db, size_t max_edges, size_t ef_cons) 
    : storage(db), M(max_edges), M_max(max_edges), M_max0(max_edges * 2), 
      ef_construction(ef_cons), max_layer(-1), entry_point(0) {
    
    level_mult = 1.0 / log(1.0 * M);
    rng.seed(1337); // Fixed seed for reproducible benchmarks
}

int HNSWGraph::get_random_level() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(rng)) * level_mult;
    return (int)r;
}

std::vector<std::pair<float, size_t>> HNSWGraph::search_layer(
    const std::vector<float>& query, size_t ep, size_t ef, int layer) const {
    
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<>> candidates;
    std::priority_queue<std::pair<float, size_t>> top_results;
    std::unordered_set<size_t> visited;

    const float* q_ptr = query.data();
    float entry_dist = storage.compute_l2_neon(q_ptr, storage.get_vector(ep), storage.get_dim());
    
    candidates.push({entry_dist, ep});
    top_results.push({entry_dist, ep});
    visited.insert(ep);

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        if (current.first > top_results.top().first && top_results.size() == ef) break;

        // Only traverse neighbors that exist on this specific layer
        for (size_t neighbor_id : graph[layer][current.second]) {
            if (visited.find(neighbor_id) == visited.end()) {
                visited.insert(neighbor_id);
                float dist = storage.compute_l2_neon(q_ptr, storage.get_vector(neighbor_id), storage.get_dim());
                
                if (top_results.size() < ef || dist < top_results.top().first) {
                    candidates.push({dist, neighbor_id});
                    top_results.push({dist, neighbor_id});
                    if (top_results.size() > ef) top_results.pop();
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
    
    // Expand tracking structures if this is a new vector
    if (new_vid >= node_levels.size()) {
        size_t new_size = new_vid + 1000;
        node_levels.resize(new_size, -1);
        
        // Also expand all existing layers in the graph to accommodate the new ID
        for (auto& layer : graph) {
            layer.resize(new_size);
        }
    }
    node_levels[new_vid] = new_level;

    // Expand graph layers if the new level is the highest we've ever seen
    while (graph.size() <= (size_t)new_level) {
        // Initialize the new layer to the current size of node_levels
        graph.push_back(std::vector<std::vector<size_t>>(node_levels.size()));
    }

    if (new_vid == 0) {
        max_layer = new_level;
        entry_point = 0;
        return;
    }

    const float* raw_vec = storage.get_vector(new_vid);
    std::vector<float> query(raw_vec, raw_vec + storage.get_dim());
    
    size_t curr_ep = entry_point;
    
    // Phase 1: Fast Search down to the new_level
    for (int lc = max_layer; lc > new_level; --lc) {
        auto nearest = search_layer(query, curr_ep, 1, lc); // ef=1 for pure greedy routing
        curr_ep = nearest[0].second;
    }

    // Phase 2: Wire the graph from new_level down to Layer 0
    for (int lc = std::min(max_layer, new_level); lc >= 0; --lc) {
        auto nearest = search_layer(query, curr_ep, ef_construction, lc);
        curr_ep = nearest[0].second; 
        
        size_t max_conn = (lc == 0) ? M_max0 : M_max;
        size_t connections = std::min(M, nearest.size());

        for (size_t i = 0; i < connections; ++i) {
            size_t neighbor_id = nearest[i].second;
            
            graph[lc][new_vid].push_back(neighbor_id);
            graph[lc][neighbor_id].push_back(new_vid);
            
            // Prune overflowing neighbors on this layer
            if (graph[lc][neighbor_id].size() > max_conn) {
                graph[lc][neighbor_id].erase(graph[lc][neighbor_id].begin());
            }
        }
    }

    // Phase 3: Update global entry point if this node reached a new height
    if (new_level > max_layer) {
        max_layer = new_level;
        entry_point = new_vid;
    }
}

std::vector<std::pair<float, size_t>> HNSWGraph::search_ann(
    const std::vector<float>& query, size_t k, size_t ef_search) const {
    
    if (max_layer == -1) return {};

    size_t curr_ep = entry_point;

    // Fast routing down to layer 1
    for (int lc = max_layer; lc > 0; --lc) {
        auto nearest = search_layer(query, curr_ep, 1, lc);
        curr_ep = nearest[0].second;
    }

    // Beam search at layer 0 for accuracy
    return search_layer(query, curr_ep, std::max(k, ef_search), 0);
}
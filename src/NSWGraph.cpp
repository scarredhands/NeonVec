#include "NSWGraph.h"
#include <algorithm>
#include <cassert>
#include <iostream>

NSWGraph::NSWGraph(VectorStorage& db, size_t max_edges, size_t ef_cons) 
    : storage(db), M(max_edges), ef_construction(ef_cons) {
    // Pre-allocate adjacency list to match storage capacity
    adjacency_list.resize(storage.get_count());
}

std::vector<std::pair<float, size_t>> NSWGraph::search_layer(
    const std::vector<float>& query, size_t entry_point, size_t ef) const {
    
    // Min-heap to explore closest candidates first
    std::priority_queue<
        std::pair<float, size_t>, 
        std::vector<std::pair<float, size_t>>, 
        std::greater<std::pair<float, size_t>>> candidates;
        
    // Max-heap to track the best 'ef' results found so far
    std::priority_queue<std::pair<float, size_t>> top_results;
    
    std::unordered_set<size_t> visited;

    const float* q_ptr = query.data();
    float entry_dist = storage.compute_l2_neon(q_ptr, storage.get_vector(entry_point), storage.get_dim());
    
    candidates.push({entry_dist, entry_point});
    top_results.push({entry_dist, entry_point});
    visited.insert(entry_point);

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        // If the closest candidate is further than the worst result we have, stop searching
        if (current.first > top_results.top().first && top_results.size() == ef) {
            break;
        }

        for (size_t neighbor_id : adjacency_list[current.second]) {
            if (visited.find(neighbor_id) == visited.end()) {
                visited.insert(neighbor_id);
                
                float dist = storage.compute_l2_neon(q_ptr, storage.get_vector(neighbor_id), storage.get_dim());
                
                if (top_results.size() < ef || dist < top_results.top().first) {
                    candidates.push({dist, neighbor_id});
                    top_results.push({dist, neighbor_id});
                    
                    if (top_results.size() > ef) {
                        top_results.pop();
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

void NSWGraph::insert(size_t new_vid) {
    if (new_vid >= adjacency_list.size()) {
        adjacency_list.resize(new_vid + 1000); 
    }

    if (new_vid == 0) {
        return; // First node has no neighbors to connect to
    }

    const float* raw_vec = storage.get_vector(new_vid);
    std::vector<float> query(raw_vec, raw_vec + storage.get_dim());

    size_t entry_point = 0; 
    auto nearest = search_layer(query, entry_point, ef_construction);

    size_t connections = std::min(M, nearest.size());
    for (size_t i = 0; i < connections; ++i) {
        size_t neighbor_id = nearest[i].second;
        
        // Connect bidirectionally
        adjacency_list[new_vid].push_back(neighbor_id);
        adjacency_list[neighbor_id].push_back(new_vid);
        
        // FIXED PRUNING LOGIC:
        // Allow up to 2*M connections for graph connectivity.
        // If it overflows, delete the OLDEST connection (at the front), not the NEWEST one!
        if (adjacency_list[neighbor_id].size() > M * 2) {
            adjacency_list[neighbor_id].erase(adjacency_list[neighbor_id].begin());
        }
    }
}

std::vector<std::pair<float, size_t>> NSWGraph::search_ann(
    const std::vector<float>& query, size_t entry_point, size_t k) const {
    return search_layer(query, entry_point, std::max(k, ef_construction));
}
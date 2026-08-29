#ifndef HNSW_GRAPH_H
#define HNSW_GRAPH_H

#include "VectorStorage.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <cstdint>

class HNSWGraph {
private:
    VectorStorage& storage;
    size_t M;
    size_t M_max;
    size_t M_max0;
    size_t ef_construction;
    
    double level_mult;
    int max_layer;
    size_t entry_point;
    
    std::mt19937 rng;
    
    std::vector<int> node_levels;
    std::vector<std::vector<std::vector<size_t>>> graph;

    int get_random_level();
    
    // UPDATED: Now uses int8_t for incredibly fast routing
    std::vector<std::pair<float, size_t>> search_layer(
        const std::vector<int8_t>& query, size_t ep, size_t ef, int layer) const;

public:
    HNSWGraph(VectorStorage& db, size_t max_edges, size_t ef_cons);
    
    void insert(size_t new_vid);
    
    // UPDATED: Parameter explicitly named query_float to prevent collisions
    std::vector<std::pair<float, size_t>> search_ann(
        const std::vector<float>& query_float, size_t k, size_t ef_search) const;
};

#endif
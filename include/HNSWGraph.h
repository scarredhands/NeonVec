#ifndef HNSW_GRAPH_H
#define HNSW_GRAPH_H

#include "VectorStorage.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <random>

class HNSWGraph {
private:
    VectorStorage& storage;
    
    // 3D Graph: graph[level][node_id] = list_of_neighbors
    std::vector<std::vector<std::vector<size_t>>> graph;
    std::vector<int> node_levels; // Tracks the max level of each node
    
    size_t M; 
    size_t M_max;
    size_t M_max0; // Layer 0 usually allows double the connections
    size_t ef_construction;
    double level_mult; // Multiplier for random level generation

    int max_layer;
    size_t entry_point;

    std::mt19937 rng; // Random number generator

    // Generates a random level using an exponentially decaying probability
    int get_random_level();

    // Beam search constrained to a specific layer
    std::vector<std::pair<float, size_t>> search_layer(
        const std::vector<float>& query, 
        size_t ep, 
        size_t ef, 
        int layer) const;

public:
    HNSWGraph(VectorStorage& db, size_t max_edges = 16, size_t ef_cons = 200);

    void insert(size_t new_vid);

    std::vector<std::pair<float, size_t>> search_ann(
        const std::vector<float>& query, 
        size_t k, 
        size_t ef_search) const;
};

#endif // HNSW_GRAPH_H
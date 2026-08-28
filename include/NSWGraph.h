#ifndef NSW_GRAPH_H
#define NSW_GRAPH_H

#include "vectorStorage.h"
#include <vector>
#include <queue>
#include <unordered_set>

class NSWGraph {
private:
    VectorStorage& storage;
    
    // Each index maps to a list of neighbor IDs
    std::vector<std::vector<size_t>> adjacency_list;
    
    // Hyperparameters
    size_t M;      // Max number of edges per node
    size_t ef_construction; // Beam width during graph insertion

    // Helper for greedy search
    std::vector<std::pair<float, size_t>> search_layer(
        const std::vector<float>& query, 
        size_t entry_point, 
        size_t ef) const;

public:
    NSWGraph(VectorStorage& db, size_t max_edges = 16, size_t ef_cons = 50);

    // Build the graph dynamically
    void insert(size_t new_vid);

    // Fast approximate search
    std::vector<std::pair<float, size_t>> search_ann(
        const std::vector<float>& query, 
        size_t entry_point, 
        size_t k) const;
};

#endif // NSW_GRAPH_H
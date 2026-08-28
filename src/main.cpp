#include "VectorStorage.h"
#include "NSWGraph.h"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    size_t dim = 128; 
    size_t num_vectors = 1000000;
    
    VectorStorage db(dim, num_vectors);
    NSWGraph index(db, 16, 50);
    
    std::cout << "Generating and inserting " << num_vectors << " synthetic vectors...\n";
    
    // Generate and insert vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec(dim);
        for (size_t d = 0; d < dim; ++d) {
            vec[d] = static_cast<float>(rand()) / RAND_MAX;
        }
        size_t id = db.add_vector(vec);
        index.insert(id);
    }
    
    // Generate a random query
    std::vector<float> query(dim);
    for (size_t d = 0; d < dim; ++d) {
        query[d] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    std::cout << "Testing Flat Search (Brute Force)...\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto exact_results = db.search_knn(query, 5);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Latency: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";

    std::cout << "\nTesting NSW Graph Search (ANN)...\n";
    start = std::chrono::high_resolution_clock::now();
    auto ann_results = index.search_ann(query, 0, 5);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Latency: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";

    std::cout << "\nTop 5 Results Comparison (ID | Distance):\n";
    for(int i = 0; i < 5; ++i) {
        std::cout << "Exact: " << exact_results[i].second << " (" << exact_results[i].first << ") "
                  << "| ANN: " << ann_results[i].second << " (" << ann_results[i].first << ")\n";
    }

    return 0;
}
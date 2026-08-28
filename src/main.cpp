#include "vectorStorage.h"
#include <iostream>
#include <vector>

int main() {
    size_t dim = 128; // Standard embedding size
    size_t max_capacity = 10000;
    
    // Initialize the storage engine
    VectorStorage db(dim, max_capacity);
    
    // Generate synthetic vector data
    std::vector<float> vec1(dim, 0.1f);
    std::vector<float> vec2(dim, 0.9f);
    
    // Insert vectors into the engine
    size_t id1 = db.add_vector(vec1);
    size_t id2 = db.add_vector(vec2);
    
    std::cout << "Added vector 1 at ID: " << id1 << std::endl;
    std::cout << "Added vector 2 at ID: " << id2 << std::endl;
    
    // Query near vec1
    std::vector<float> query(dim, 0.9f); 
    auto results = db.search_knn(query, 1);
    
    std::cout << "\nNearest neighbor to query:\n";
    for (const auto& res : results) {
        std::cout << "ID: " << res.second << ", L2 Distance (Squared): " << res.first << std::endl;
    }
    
    return 0;
}
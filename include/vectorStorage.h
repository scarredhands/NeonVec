#ifndef VECTOR_STORAGE_H
#define VECTOR_STORAGE_H

#include <vector>
#include <cstddef>
#include <utility>

class VectorStorage {
private:
    float* data;
    size_t dim;
    size_t capacity;
    size_t count;

public:
    // Constructor and Destructor
    VectorStorage(size_t dimension, size_t max_vectors);
    ~VectorStorage();

    // Core Engine Operations
    size_t add_vector(const std::vector<float>& vec);
    const float* get_vector(size_t id) const;
    
    // Getters
    size_t get_count() const;
    size_t get_dim() const;

    // Hardware Accelerated Kernels
    static float compute_l2_neon(const float* a, const float* b, size_t dim);
    static float compute_dot_neon(const float* a, const float* b, size_t dim);

    // Search Algorithm
    std::vector<std::pair<float, size_t>> search_knn(const std::vector<float>& query, int k) const;
};

#endif // VECTOR_STORAGE_H
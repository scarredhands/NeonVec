#include "VectorStorage.h"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <algorithm>

VectorStorage::VectorStorage(size_t dim, size_t capacity)
    : dim(dim), max_capacity(capacity), count(0) {
    // Allocate RAM for 8-bit integers
    data = new int8_t[max_capacity * dim];
}

VectorStorage::~VectorStorage() {
    delete[] data;
}

std::vector<int8_t> VectorStorage::quantize(const std::vector<float>& vec) {
    std::vector<int8_t> q_vec(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        // Clip values to [-1.0, 1.0] and scale to 8-bit integer [-127, 127]
        float clipped = std::max(-1.0f, std::min(1.0f, vec[i]));
        q_vec[i] = static_cast<int8_t>(clipped * 127.0f);
    }
    return q_vec;
}

size_t VectorStorage::add_vector(const std::vector<float>& vec) {
    if (count >= max_capacity) throw std::runtime_error("Storage full");
    
    // Quantize on ingestion
    std::vector<int8_t> q_vec = quantize(vec);
    
    // Copy the compressed 8-bit data into the main memory block
    std::memcpy(data + (count * dim), q_vec.data(), dim * sizeof(int8_t));
    return count++;
}

float VectorStorage::compute_l2_sq(const int8_t* vec1, const int8_t* vec2, size_t dim) const {
    int32_t dist = 0; 
    for (size_t i = 0; i < dim; ++i) {
        int32_t diff = vec1[i] - vec2[i];
        dist += diff * diff; // Integer multiplication is blazingly fast
    }
    // Return as float because HNSW priority queues expect float distances
    return static_cast<float>(dist); 
}
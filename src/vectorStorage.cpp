#include "vectorStorage.h"
#include <arm_neon.h> // ARM SIMD hardware acceleration
#include <cstdlib>
#include <cassert>
#include <queue>
#include <algorithm>
#include <stdexcept>

// Task 1: 64-byte aligned memory allocation
VectorStorage::VectorStorage(size_t dimension, size_t max_vectors) 
    : dim(dimension), capacity(max_vectors), count(0) {
    
    // posix_memalign ensures the memory aligns perfectly with CPU cache lines
    if (posix_memalign((void**)&data, 64, capacity * dim * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
}

VectorStorage::~VectorStorage() {
    free(data);
}

size_t VectorStorage::add_vector(const std::vector<float>& vec) {
    assert(vec.size() == dim && "Vector dimension mismatch!");
    assert(count < capacity && "Storage capacity reached!");
    
    size_t id = count;
    // Copy the vector into the contiguous flat buffer
    for (size_t i = 0; i < dim; ++i) {
        data[id * dim + i] = vec[i];
    }
    count++;
    return id;
}

const float* VectorStorage::get_vector(size_t id) const {
    return &data[id * dim];
}

size_t VectorStorage::get_count() const { return count; }
size_t VectorStorage::get_dim() const { return dim; }

// Task 2: SIMD Distance Kernels 
float VectorStorage::compute_l2_neon(const float* a, const float* b, size_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    // Process 4 floats per cycle
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum_vec = vmlaq_f32(sum_vec, diff, diff); // sum += diff * diff
    }
    
    // Extract SIMD results back to scalar
    float result[4];
    vst1q_f32(result, sum_vec);
    float total_distance = result[0] + result[1] + result[2] + result[3];
    
    // Handle remainder if dim is not a multiple of 4
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        total_distance += diff * diff;
    }
    
    return total_distance;
}

float VectorStorage::compute_dot_neon(const float* a, const float* b, size_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum_vec = vmlaq_f32(sum_vec, va, vb);
    }
    
    float result[4];
    vst1q_f32(result, sum_vec);
    float total = result[0] + result[1] + result[2] + result[3];
    
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// Task 3: Flat Baseline (Brute-force k-NN Search)
std::vector<std::pair<float, size_t>> VectorStorage::search_knn(const std::vector<float>& query, int k) const {
    assert(query.size() == dim && "Query dimension mismatch!");
    
    // Max-heap to keep track of the closest 'k' vectors
    std::priority_queue<std::pair<float, size_t>> max_heap; 
    
    const float* q_ptr = query.data();

    for (size_t i = 0; i < count; ++i) {
        float dist = compute_l2_neon(q_ptr, get_vector(i), dim);
        
        if (max_heap.size() < (size_t)k) {
            max_heap.push({dist, i});
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.push({dist, i});
        }
    }
    
    // Extract from heap and reverse to get smallest distance first
    std::vector<std::pair<float, size_t>> results;
    while (!max_heap.empty()) {
        results.push_back(max_heap.top());
        max_heap.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}
#ifndef VECTOR_STORAGE_H
#define VECTOR_STORAGE_H

#include <vector>
#include <cstdint>

class VectorStorage {
private:
    size_t dim;
    size_t max_capacity;
    size_t count;
    
    // 75% MEMORY REDUCTION: Storing 8-bit ints instead of 32-bit floats
    int8_t* data; 

public:
    VectorStorage(size_t dim, size_t capacity);
    ~VectorStorage();

    // Quantization Engine
    static std::vector<int8_t> quantize(const std::vector<float>& vec);

    size_t add_vector(const std::vector<float>& vec);
    
    // Returns quantized vector
    const int8_t* get_vector(size_t id) const { return data + (id * dim); }
    
    size_t get_count() const { return count; }
    size_t get_dim() const { return dim; }

    // Computes L2 distance purely using integers
    float compute_l2_sq(const int8_t* vec1, const int8_t* vec2, size_t dim) const;
};

#endif
#ifndef VECTOR_STORAGE_H
#define VECTOR_STORAGE_H

#include <vector>
#include <cstdint>
#include <fstream>

class VectorStorage {
private:
    size_t dim;
    size_t max_capacity;
    size_t count;
    int8_t* data; 
    std::vector<uint8_t> tombstones; 

public:
    VectorStorage(size_t dim, size_t capacity);
    ~VectorStorage();

    static std::vector<int8_t> quantize(const std::vector<float>& vec);
    size_t add_vector(const std::vector<float>& vec);
    void mark_deleted(size_t id);
    bool is_deleted(size_t id) const;

    const int8_t* get_vector(size_t id) const { return data + (id * dim); }
    size_t get_count() const { return count; }
    size_t get_dim() const { return dim; }

    float compute_l2_sq(const int8_t* vec1, const int8_t* vec2, size_t dim) const;

    // SNAPSHOT API
    void save(std::ostream& out) const;
    void load(std::istream& in);
};

#endif
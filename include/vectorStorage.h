// #ifndef VECTOR_STORAGE_H
// #define VECTOR_STORAGE_H

// #include <vector>
// #include <cstdint>
// #include <fstream>

// class VectorStorage {
// private:
//     size_t dim;
//     size_t max_capacity;
//     size_t count;
//     int8_t* data; 
//     std::vector<uint8_t> tombstones; 

// public:
//     VectorStorage(size_t dim, size_t capacity);
//     ~VectorStorage();

//     static std::vector<int8_t> quantize(const std::vector<float>& vec);
//     size_t add_vector(const std::vector<float>& vec);
//     void mark_deleted(size_t id);
//     bool is_deleted(size_t id) const;

//     const int8_t* get_vector(size_t id) const { return data + (id * dim); }
//     size_t get_count() const { return count; }
//     size_t get_dim() const { return dim; }

//     float compute_l2_sq(const int8_t* vec1, const int8_t* vec2, size_t dim) const;

//     // SNAPSHOT API
//     void save(std::ostream& out) const;
//     void load(std::istream& in);
// };

// #endif


#ifndef VECTOR_STORAGE_H
#define VECTOR_STORAGE_H

#include <vector>
#include <cstdint>
#include <fstream>

class VectorStorage {
private:
    size_t dim;
    size_t count;
    size_t chunk_size; // Formerly max_capacity
    
    // SEGMENTED MEMORY: A list of pointers to fixed-size RAM blocks
    std::vector<int8_t*> chunks; 
    
    // Tombstones will now dynamically grow using push_back (1 byte per vector is harmless to resize)
    std::vector<uint8_t> tombstones; 

public:
    VectorStorage(size_t dim, size_t chunk_size);
    ~VectorStorage();

    static std::vector<int8_t> quantize(const std::vector<float>& vec);
    size_t add_vector(const std::vector<float>& vec);
    void mark_deleted(size_t id);
    bool is_deleted(size_t id) const;

    // THE 2D POINTER LOOKUP
    const int8_t* get_vector(size_t id) const { 
        size_t chunk_idx = id / chunk_size;
        size_t offset = id % chunk_size;
        return chunks[chunk_idx] + (offset * dim); 
    }
    
    size_t get_count() const { return count; }
    size_t get_dim() const { return dim; }

    float compute_l2_sq(const int8_t* vec1, const int8_t* vec2, size_t dim) const;

    void save(std::ostream& out) const;
    void load(std::istream& in);
};

#endif

// #include "VectorStorage.h"
// #include <cmath>
// #include <stdexcept>
// #include <cstring>
// #include <algorithm>

// VectorStorage::VectorStorage(size_t dim, size_t capacity)
//     : dim(dim), max_capacity(capacity), count(0) {
//     data = new int8_t[max_capacity * dim];
//     tombstones.resize(max_capacity, 0); 
// }

// VectorStorage::~VectorStorage() {
//     delete[] data;
// }

// std::vector<int8_t> VectorStorage::quantize(const std::vector<float>& vec) {
//     std::vector<int8_t> q_vec(vec.size());
//     for (size_t i = 0; i < vec.size(); ++i) {
//         float clipped = std::max(-1.0f, std::min(1.0f, vec[i]));
//         q_vec[i] = static_cast<int8_t>(clipped * 127.0f);
//     }
//     return q_vec;
// }

// size_t VectorStorage::add_vector(const std::vector<float>& vec) {
//     if (count >= max_capacity) throw std::runtime_error("Storage full");
//     std::vector<int8_t> q_vec = quantize(vec);
//     std::memcpy(data + (count * dim), q_vec.data(), dim * sizeof(int8_t));
//     tombstones[count] = 0; 
//     return count++;
// }

// void VectorStorage::mark_deleted(size_t id) {
//     if (id < count) tombstones[id] = 1; 
// }

// bool VectorStorage::is_deleted(size_t id) const {
//     return (id < count) && (tombstones[id] == 1);
// }

// float VectorStorage::compute_l2_sq(const int8_t* vec1, const int8_t* vec2, size_t dim) const {
//     int32_t dist = 0; 
//     for (size_t i = 0; i < dim; ++i) {
//         int32_t diff = vec1[i] - vec2[i];
//         dist += diff * diff; 
//     }
//     return static_cast<float>(dist); 
// }

// // --- SNAPSHOT LOGIC ---
// void VectorStorage::save(std::ostream& out) const {
//     out.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
//     out.write(reinterpret_cast<const char*>(data), count * dim * sizeof(int8_t));
//     out.write(reinterpret_cast<const char*>(tombstones.data()), count * sizeof(uint8_t));
// }

// void VectorStorage::load(std::istream& in) {
//     in.read(reinterpret_cast<char*>(&count), sizeof(size_t));
//     in.read(reinterpret_cast<char*>(data), count * dim * sizeof(int8_t));
//     in.read(reinterpret_cast<char*>(tombstones.data()), count * sizeof(uint8_t));
// }

#include "VectorStorage.h"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <algorithm>

VectorStorage::VectorStorage(size_t dim, size_t chunk_size)
    : dim(dim), count(0), chunk_size(chunk_size) {
    // Allocate the very first chunk on startup
    chunks.push_back(new int8_t[chunk_size * dim]);
}

VectorStorage::~VectorStorage() {
    // Free all dynamically allocated chunks
    for (int8_t* chunk : chunks) {
        delete[] chunk;
    }
}

std::vector<int8_t> VectorStorage::quantize(const std::vector<float>& vec) {
    std::vector<int8_t> q_vec(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        float clipped = std::max(-1.0f, std::min(1.0f, vec[i]));
        q_vec[i] = static_cast<int8_t>(clipped * 127.0f);
    }
    return q_vec;
}

size_t VectorStorage::add_vector(const std::vector<float>& vec) {
    // 1. Calculate which chunk this vector belongs in
    size_t chunk_idx = count / chunk_size;
    
    // 2. DYNAMIC RESIZING: If we need a new chunk, allocate it!
    // Because CoreEngine holds a unique_lock during insert, this is 100% thread-safe.
    if (chunk_idx >= chunks.size()) {
        chunks.push_back(new int8_t[chunk_size * dim]);
    }
    
    size_t offset = count % chunk_size;
    std::vector<int8_t> q_vec = quantize(vec);
    
    // 3. Write directly to the correct chunk in RAM
    std::memcpy(chunks[chunk_idx] + (offset * dim), q_vec.data(), dim * sizeof(int8_t));
    
    tombstones.push_back(0); // Dynamically add an active tombstone flag
    return count++;
}

void VectorStorage::mark_deleted(size_t id) {
    if (id < count) tombstones[id] = 1; 
}

bool VectorStorage::is_deleted(size_t id) const {
    return (id < count) && (tombstones[id] == 1);
}

float VectorStorage::compute_l2_sq(const int8_t* vec1, const int8_t* vec2, size_t dim) const {
    int32_t dist = 0; 
    for (size_t i = 0; i < dim; ++i) {
        int32_t dist_diff = vec1[i] - vec2[i];
        dist += dist_diff * dist_diff; 
    }
    return static_cast<float>(dist); 
}

// --- SNAPSHOT LOGIC ---
void VectorStorage::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
    
    // Write all chunks sequentially so the disk file is contiguous
    size_t vectors_written = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        size_t vectors_in_chunk = std::min(chunk_size, count - vectors_written);
        out.write(reinterpret_cast<const char*>(chunks[i]), vectors_in_chunk * dim * sizeof(int8_t));
        vectors_written += vectors_in_chunk;
    }
    
    out.write(reinterpret_cast<const char*>(tombstones.data()), count * sizeof(uint8_t));
}

void VectorStorage::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&count), sizeof(size_t));
    
    // Destroy default chunk
    for (int8_t* chunk : chunks) delete[] chunk;
    chunks.clear();
    
    // Calculate required chunks and reconstruct memory
    size_t num_chunks = (count + chunk_size - 1) / chunk_size;
    size_t vectors_read = 0;
    
    for (size_t i = 0; i < num_chunks; ++i) {
        chunks.push_back(new int8_t[chunk_size * dim]);
        size_t vectors_in_chunk = std::min(chunk_size, count - vectors_read);
        in.read(reinterpret_cast<char*>(chunks[i]), vectors_in_chunk * dim * sizeof(int8_t));
        vectors_read += vectors_in_chunk;
    }
    
    tombstones.resize(count);
    if (count > 0) {
        in.read(reinterpret_cast<char*>(tombstones.data()), count * sizeof(uint8_t));
    }
}

#include "CoreEngine.h"
#include <iostream>
#include <stdexcept>

// Initialize HNSW with dense parameters: M = 32, ef_construction = 200
CoreEngine::CoreEngine(size_t dim, size_t capacity, const std::string& log_file)
    : storage(dim, capacity), index(storage, 32, 200), wal_path(log_file) {
    
    // Open the WAL file in Append mode and Binary mode
    wal_writer.open(wal_path, std::ios::app | std::ios::binary);
    if (!wal_writer.is_open()) {
        throw std::runtime_error("Failed to open Write-Ahead Log.");
    }
}

CoreEngine::~CoreEngine() {
    if (wal_writer.is_open()) {
        wal_writer.close();
    }
}

void CoreEngine::log_insert_to_disk(size_t id) {
    // 1. Write the Vector ID (8 bytes)
    wal_writer.write(reinterpret_cast<const char*>(&id), sizeof(size_t));
    
    // 2. Write the compressed 8-bit int data directly from RAM to SSD (dim * 1 byte)
    const int8_t* compressed_vec = storage.get_vector(id);
    wal_writer.write(reinterpret_cast<const char*>(compressed_vec), storage.get_dim() * sizeof(int8_t));
    
    // 3. Force the OS to write to physical disk immediately
    wal_writer.flush(); 
}

size_t CoreEngine::insert(const std::vector<float>& vec) {
    // UNIQUE LOCK: Blocks all other readers and writers while modifying the DB
    std::unique_lock<std::shared_mutex> lock(rw_lock);
    
    size_t id = storage.add_vector(vec);
    index.insert(id);
    
    log_insert_to_disk(id);
    
    return id;
}

std::vector<std::pair<float, size_t>> CoreEngine::search(const std::vector<float>& query, size_t k) const {
    // SHARED LOCK: Allows multiple concurrent searches, but blocks any new inserts
    std::shared_lock<std::shared_mutex> lock(rw_lock);
    
    if (storage.get_count() == 0) {
        return {};
    }
    
    // Pass query through HNSW (ef_search = 100 for high recall)
    auto results = index.search_ann(query, k, 100);
    
    // Slice results to match requested 'k' precisely
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

void CoreEngine::recover_from_wal() {
    // Open file in Read mode and Binary mode
    std::ifstream wal_reader(wal_path, std::ios::in | std::ios::binary);
    if (!wal_reader.is_open()) return;

    size_t id;
    size_t dim = storage.get_dim();
    std::vector<int8_t> compressed_vec(dim);
    size_t recovered_count = 0;

    // Read sequentially until the end of the file
    while (wal_reader.read(reinterpret_cast<char*>(&id), sizeof(size_t))) {
        wal_reader.read(reinterpret_cast<char*>(compressed_vec.data()), dim * sizeof(int8_t));
        
        // Re-insert into storage and graph indexing layers
        // (Note: To optimize recovery further, you can add an internal add_compressed_vector method to VectorStorage)
        std::vector<float> float_reconstruction(dim);
        for(size_t i = 0; i < dim; ++i) {
            float_reconstruction[i] = static_cast<float>(compressed_vec[i]) / 127.0f;
        }
        
        storage.add_vector(float_reconstruction);
        index.insert(id);
        recovered_count++;
    }
    
    std::cout << "Recovered " << recovered_count << " vectors from WAL." << std::endl;
}
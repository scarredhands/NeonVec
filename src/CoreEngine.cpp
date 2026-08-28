#include "CoreEngine.h"
#include <iostream>
#include <stdexcept>

// Initialize HNSW with dense parameters: M = 32, ef_construction = 200
CoreEngine::CoreEngine(size_t dim, size_t capacity, const std::string& log_file)
    : storage(dim, capacity), index(storage, 32, 200), wal_path(log_file) {
    
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

void CoreEngine::log_insert_to_disk(size_t id, const std::vector<float>& vec) {
    wal_writer.write(reinterpret_cast<const char*>(&id), sizeof(size_t));
    wal_writer.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    wal_writer.flush(); 
}

size_t CoreEngine::insert(const std::vector<float>& vec) {
    std::unique_lock<std::shared_mutex> lock(rw_lock);
    
    size_t id = storage.add_vector(vec);
    index.insert(id);
    
    log_insert_to_disk(id, vec);
    return id;
}

std::vector<std::pair<float, size_t>> CoreEngine::search(const std::vector<float>& query, size_t k) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock);
    
    if (storage.get_count() == 0) return {};
    
    // HNSW requires 'ef_search' at query time. We set it to 100 for high recall.
    return index.search_ann(query, k, 100); 
}

std::vector<std::pair<float, size_t>> CoreEngine::search_exact(const std::vector<float>& query, size_t k) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock);
    
    if (storage.get_count() == 0) return {};
    return storage.search_knn(query, k);
}

void CoreEngine::recover_from_wal() {
    std::ifstream wal_reader(wal_path, std::ios::in | std::ios::binary);
    if (!wal_reader.is_open()) return;

    size_t id;
    std::vector<float> vec(storage.get_dim());
    size_t recovered_count = 0;

    while (wal_reader.read(reinterpret_cast<char*>(&id), sizeof(size_t))) {
        wal_reader.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(float));
        
        storage.add_vector(vec);
        index.insert(id);
        recovered_count++;
    }
    std::cout << "Recovered " << recovered_count << " vectors from WAL." << std::endl;
}
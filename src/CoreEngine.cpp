#include "CoreEngine.h"
#include <iostream>
#include <chrono>

CoreEngine::CoreEngine(size_t dim, size_t capacity, const std::string& log_file)
    : storage(dim, capacity), index(storage, 32, 200), wal_path(log_file), stop_worker(false) {
    
    wal_writer.open(wal_path, std::ios::app | std::ios::binary);
    
    // Launch the background thread!
    bg_worker = std::thread(&CoreEngine::maintenance_loop, this);
}

CoreEngine::~CoreEngine() {
    // 1. Tell the background thread to stop
    stop_worker = true;
    
    // 2. Wait for it to finish its current loop before destroying memory (CRITICAL)
    if (bg_worker.joinable()) {
        bg_worker.join();
    }
    if (wal_writer.is_open()) {
        wal_writer.close();
    }
}

// THE BACKGROUND THREAD
void CoreEngine::maintenance_loop() {
    while (!stop_worker) {
        // Sleep for 10 seconds, then wake up and prune the graph
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        // Grab a UNIQUE lock because we are physically altering graph edges
        std::unique_lock<std::shared_mutex> lock(rw_lock);
        index.prune_tombstones();
    }
}

void CoreEngine::log_operation_to_disk(uint8_t op_code, size_t id) {
    wal_writer.write(reinterpret_cast<const char*>(&op_code), sizeof(uint8_t));
    wal_writer.write(reinterpret_cast<const char*>(&id), sizeof(size_t));
    
    if (op_code == 0) { // If it's an insert, write the 8-bit vector payload too
        const int8_t* compressed_vec = storage.get_vector(id);
        wal_writer.write(reinterpret_cast<const char*>(compressed_vec), storage.get_dim() * sizeof(int8_t));
    }
    wal_writer.flush(); 
}

size_t CoreEngine::insert(const std::vector<float>& vec) {
    std::unique_lock<std::shared_mutex> lock(rw_lock);
    size_t id = storage.add_vector(vec);
    index.insert(id);
    log_operation_to_disk(0, id); // OpCode 0 = Insert
    return id;
}

void CoreEngine::delete_vector(size_t id) {
    std::unique_lock<std::shared_mutex> lock(rw_lock);
    storage.mark_deleted(id);
    log_operation_to_disk(1, id); // OpCode 1 = Delete
}

std::vector<std::pair<float, size_t>> CoreEngine::search(const std::vector<float>& query, size_t k) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock);
    if (storage.get_count() == 0) return {};
    
    auto results = index.search_ann(query, k, 100);
    if (results.size() > k) results.resize(k);
    return results;
}

void CoreEngine::recover_from_wal() {
    std::ifstream wal_reader(wal_path, std::ios::in | std::ios::binary);
    if (!wal_reader.is_open()) return;

    uint8_t op_code;
    size_t id;
    size_t dim = storage.get_dim();
    std::vector<int8_t> compressed_vec(dim);

    while (wal_reader.read(reinterpret_cast<char*>(&op_code), sizeof(uint8_t))) {
        wal_reader.read(reinterpret_cast<char*>(&id), sizeof(size_t));
        
        if (op_code == 0) { // Replay Insert
            wal_reader.read(reinterpret_cast<char*>(compressed_vec.data()), dim * sizeof(int8_t));
            std::vector<float> float_reconstruction(dim);
            for(size_t i = 0; i < dim; ++i) {
                float_reconstruction[i] = static_cast<float>(compressed_vec[i]) / 127.0f;
            }
            storage.add_vector(float_reconstruction);
            index.insert(id);
        } else if (op_code == 1) { // Replay Delete
            storage.mark_deleted(id);
        }
    }
}
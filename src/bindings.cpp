#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "CoreEngine.h"

namespace py = pybind11;

PYBIND11_MODULE(neonvec, m) {
    m.doc() = "NeonVec: High-performance HNSW vector database engine";

    py::class_<CoreEngine>(m, "CoreEngine")
        // Constructor
        .def(py::init<size_t, size_t, const std::string&>(), 
             py::arg("dim"), py::arg("capacity"), py::arg("log_file"))
        
        // Core DB Operations
        .def("insert", &CoreEngine::insert, 
             py::arg("vec"), "Insert a vector into the database")
             
        .def("delete", &CoreEngine::delete_vector, 
             py::arg("id"), "Soft delete a vector by ID")
             
        .def("search", &CoreEngine::search, 
             py::arg("query"), py::arg("k"), "Search for k nearest neighbors")
             
        // Snapshot & Compaction API
        .def("save_snapshot", &CoreEngine::save_snapshot, 
             "Save graph state and truncate WAL")
             
        .def("load_state", &CoreEngine::load_state, 
             "Load snapshot and replay recent WAL"); // <-- Notice the semicolon is ONLY here!
}
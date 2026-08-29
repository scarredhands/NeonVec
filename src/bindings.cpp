#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Magic header: Auto-converts Python lists to std::vector
#include "CoreEngine.h"

namespace py = pybind11;

// "neonvec" is the name of the module you will import in Python
PYBIND11_MODULE(neonvec, m) {
    m.doc() = "NeonVec: High-performance HNSW vector database engine";

    // Bind the CoreEngine class
    py::class_<CoreEngine>(m, "CoreEngine")
        // Bind the constructor
        .def(py::init<size_t, size_t, const std::string&>(), 
             py::arg("dim"), py::arg("capacity"), py::arg("log_file"))
        
        // Bind the core methods
        .def("insert", &CoreEngine::insert, 
             py::arg("vec"), 
             "Insert a vector into the database")
             
        .def("search", &CoreEngine::search, 
             py::arg("query"), py::arg("k"), 
             "Search for k nearest neighbors")
             
        .def("recover_from_wal", &CoreEngine::recover_from_wal, 
             "Load data from the physical disk log");
}
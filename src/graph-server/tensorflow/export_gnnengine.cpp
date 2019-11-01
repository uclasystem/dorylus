#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include <string>

#include "../engine/engine.hpp"

void
Engine::pyInit(boost::python::list argv) {
    unsigned argc = len(argv);
    char** argvCBuf = new char* [argc];
    std::string *strBuf = new std::string [argc];

    for (unsigned i = 0; i < argc; ++i) {
        strBuf[i] = boost::python::extract<std::string>(argv[i]);
        argvCBuf[i] = strBuf[i].c_str();
    }
    init(argc, argv);

    delete[] strBuf;
    delete[] argvCBuf;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(gnnengine) {
    class_<Engine>("Engine")
        .def("init", &Engine::pyInit)
        .def("output", &Engine::output)
        .def("isGPUEnabled", &Engine::isGPUEnabled)
        .def("destroy", &Engine::destroy);
};
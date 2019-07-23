#ifndef __VERTEX_PROGRAM_HPP__
#define __VERTEX_PROGRAM_HPP__


#include "vertex.hpp"


/**
 *
 * Class of a vertex program. User needs to inherit this base class in their benchmarks.
 * 
 */
class VertexProgram {

public:

    virtual void beforeIteration(unsigned layer) { }

    virtual void update(Vertex& vertex, unsigned layer) {
        assert(false);
    }

    virtual void processVertex(Vertex& vertex) {
        assert(false);
    }

    virtual void afterIteration(unsigned layer) { }
};


#endif // __VERTEX_PROGRAM_HPP__ 

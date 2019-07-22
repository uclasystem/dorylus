#ifndef __VERTEXPROGRAM_HPP__
#define __VERTEXPROGRAM_HPP__


#include "vertex.cpp"


/**
 *
 * Class of a vertex program. User needs to inherit this base class in their benchmarks.
 * 
 */
template<typename VertexType, typename EdgeType>
class VertexProgram {

public:

    virtual void beforeIteration(unsigned layer) { }

    virtual void update(Vertex<VertexType, EdgeType>& vertex, unsigned layer) {
        assert(false);
    }

    virtual void processVertex(Vertex<VertexType, EdgeType>& vertex) {
        assert(false);
    }

    virtual void afterIteration(unsigned layer) { }
};


#endif /* __VERTEXPROGRAM_HPP__ */

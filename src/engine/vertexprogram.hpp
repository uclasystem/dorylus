#ifndef __VERTEXPROGRAM_HPP__
#define __VERTEXPROGRAM_HPP__

#include "vertex.cpp"

template<typename VertexType, typename EdgeType>
class VertexProgram {

public:

    virtual void beforeIteration(unsigned layer) { }

    virtual bool update(Vertex<VertexType, EdgeType>& vertex, unsigned layer) {
        fprintf(stderr, "base update called.\n");
        assert(false);
        return false;
    }

    virtual void processVertex(Vertex<VertexType, EdgeType>& vertex) {
        fprintf(stderr, "base processVertex called.\n");
        assert(false);
    }

    virtual void afterIteration(unsigned layer) { }
};


#endif /* __VERTEXPROGRAM_HPP__ */

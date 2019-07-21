#ifndef __LIB_VERTEXPROGRAM_HPP__
#define __LIB_VERTEXPROGRAM_HPP__

#include "enginecontext.hpp"
#include "vertex.hpp"

template<typename VertexType, typename EdgeType>
class VertexProgram {
    public:
        virtual ~VertexProgram() {
        }

        virtual void beforeIteration(EngineContext& engineContext) {
        }

        virtual bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
            fprintf(stderr, "base update called.\n");
            assert(false);
            return false;
        }

        virtual void processVertex(Vertex<VertexType, EdgeType>& vertex) {
            fprintf(stderr, "base processVertex called.\n");
            assert(false);
        }


        virtual void afterIteration(EngineContext& engineContext) {
        }
};

#endif /* __LIB_VERTEXPROGRAM_HPP__ */


#ifndef __GHOST_VERTEX_HPP__
#define __GHOST_VERTEX_HPP__

#include <cassert>

template <typename VertexType>
class GhostVertex {
    public:
        
        VertexType vertexData;

        //char version;
        RWLock lock;
        int32_t degree;

        std::vector<IdType> outEdges;

        GhostVertex() {
            lock.init();
            degree = 0;
        }

        GhostVertex(const VertexType vData) : vertexData(vData) {
            lock.init();
            degree = 0;
        }

        ~GhostVertex() {
            lock.destroy();
        }

        VertexType data() {
            VertexType vData;
            lock.readLock();
            vData = vertexData.back();
            lock.unlock();
            return vData;
        }

        void setData(VertexType* value) {
            lock.writeLock();
            vertexData = *value;
            lock.unlock();
        }



	void incrementDegree() { ++degree; }
};

#endif //__GHOST_VERTEX_HPP__

#ifndef __GHOST_VERTEX_HPP__
#define __GHOST_VERTEX_HPP__

#include <cassert>

template <typename VertexType>
class GhostVertex {
    public:
        
        std::vector<VertexType> vertexData;

        //char version;
        RWLock lock;
        int32_t degree;

        std::vector<IdType> outEdges;

        GhostVertex() {
            lock.init();
            degree = 0;
        }

        GhostVertex(const VertexType vData) {
            lock.init();
            degree = 0;
            vertexData.push_back(vData);
        }

        ~GhostVertex() {
            lock.destroy();
        }

        VertexType data() {
            VertexType vData;
            lock.readLock();
            VertexType vDataRef = vertexData.back();
            copy(vDataRef.begin(), vDataRef.end(), back_inserter(vData)); 
            lock.unlock();
            return vData;
        }

        void setData(VertexType* value) {
            lock.writeLock();
            vertexData.back() = *value;
            lock.unlock();
        }

        void addData(VertexType* value) {
            lock.writeLock();
            vertexData.push_back(*value);
            lock.unlock();
        }

        void incrementDegree() { ++degree; }
};

#endif //__GHOST_VERTEX_HPP__

#ifndef __GHOST_VERTEX_HPP__
#define __GHOST_VERTEX_HPP__

#include <cassert>

template <typename VertexType>
class GhostVertex {
    public:
        
        // Use a vector to make data in old iterations persistent.
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
            vertexData.clear();
            vertexData.push_back(vData);
        }

        ~GhostVertex() {
            lock.destroy();
        }

        VertexType data() {     // Get the current value.
            // VertexType vData;
            // lock.readLock();
            // VertexType vDataRef = vertexData.back();
            // copy(vDataRef.begin(), vDataRef.end(), back_inserter(vData)); 
            // lock.unlock();
            // return vData;
            lock.readLock();
            VertexType vData = vertexData.back();
            lock.unlock();
            return vData;
        }

        void setData(VertexType* value) {   // Modify the current value.
            lock.writeLock();
            vertexData.back() = *value;
            lock.unlock();
        }

        void addData(VertexType* value) {   // Add a new value of the new iteration.
            lock.writeLock();
            vertexData.push_back(*value);
            lock.unlock();
        }

        void incrementDegree() { ++degree; }
};

#endif //__GHOST_VERTEX_HPP__

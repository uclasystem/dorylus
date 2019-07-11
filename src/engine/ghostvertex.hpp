#ifndef __GHOST_VERTEX_HPP__
#define __GHOST_VERTEX_HPP__

#include <cassert>

template <typename VertexType>
class GhostVertex {
    public:
        VertexType vData;
        //char version;
        RWLock lock;
	int32_t degree;

        std::vector<IdType> outEdges;

        GhostVertex() {
          lock.init();
	  degree = 0;
        }

        GhostVertex(const VertexType vDat) : vData(vDat) { //, version(0) {
            lock.init();
	    degree = 0;
        }

        ~GhostVertex() {
            lock.destroy();
        }

        VertexType data() {
            VertexType ret;
            lock.readLock();
            ret = vData;
            lock.unlock();
            return ret;
        }

        void setData(VertexType* value) { //, char ver) {
            //bool ret = false;
            lock.writeLock();
            //assert(version != 255);
            //if(ver > version) {
                vData = *value;
                //version = ver;
                //ret = true;
            /*} else {
                fprintf(stderr, "Dropping value %u because it is older than value %u -- because ver = %u is <= version = %u\n", *value, vData, (unsigned) ver, (unsigned) version);
            }*/
            lock.unlock();
            //return ret;
        }

	void incrementDegree() { ++degree; }
};

#endif //__GHOST_VERTEX_HPP__

#ifndef __EDGE_H__
#define __EDGE_H__

#include "../utils/utils.h"

#define LOCAL_EDGE_TYPE 'L'
#define REMOTE_EDGE_TYPE 'R'

typedef char EdgeLocationType;

template<typename EdgeType>
class Edge {
    public:
        Edge();
        Edge(IdType oId, EdgeLocationType eLocation, EdgeType eData = EdgeType());
        EdgeType data();
        EdgeLocationType getEdgeLocation();
        void setEdgeLocation(EdgeLocationType eLoc);

    protected:
        IdType otherId;
        EdgeType edgeData;
        EdgeLocationType edgeLocation;
};

template<typename EdgeType>
class InEdge: public Edge<EdgeType> {
    public:
        InEdge();
        InEdge(IdType sId, EdgeLocationType eLocation, EdgeType eData = EdgeType());
        IdType sourceId();
        void setSourceId(IdType sId);
};

template<typename EdgeType>
class OutEdge: public Edge<EdgeType> {
    public:
        OutEdge();
        OutEdge(IdType dId, EdgeLocationType eLocation, EdgeType eData = EdgeType());
        IdType destId();
        void setDestId(IdType dId);
};

#endif /* __EDGE_H__ */

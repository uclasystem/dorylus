#ifndef __EDGE_HPP__
#define __EDGE_HPP__


#include "../utils/utils.hpp"


/** Use a single byte for edge location type. */
typedef char EdgeLocationType;
#define LOCAL_EDGE_TYPE  'L'
#define REMOTE_EDGE_TYPE 'R'


/**
 *
 * Base class of a directed edge in the graph. Inherited by in / out-coming edge.
 * 
 */
template<typename EdgeType>
class Edge {

public:

    Edge(IdType oId, EdgeLocationType eLocation, EdgeType eData = EdgeType());

    EdgeType data();
    EdgeLocationType getEdgeLocation();

    void setData(EdgeType value);
    void setEdgeLocation(EdgeLocationType eLoc);

protected:

    IdType otherId;     // Id stores local_edge ? local_vid : global_vid.
    EdgeType edgeData;
    EdgeLocationType edgeLocation;
};


/**
 *
 * Class of an incoming edge.
 * 
 */
template<typename EdgeType>
class InEdge: public Edge<EdgeType> {

public:

    InEdge(IdType sId, EdgeLocationType eLocation, EdgeType eData = EdgeType());

    IdType sourceId();
    void setSourceId(IdType sId);
};


/**
 *
 * Class of an outcoming edge.
 * 
 */
template<typename EdgeType>
class OutEdge: public Edge<EdgeType> {

public:

    OutEdge(IdType dId, EdgeLocationType eLocation, EdgeType eData = EdgeType());
    
    IdType destId();
    void setDestId(IdType dId);
};


#endif /* __EDGE_HPP__ */

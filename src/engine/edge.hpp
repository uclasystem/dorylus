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
class Edge {

public:

    Edge(IdType oId, EdgeLocationType eLocation, EdgeType eData = EdgeType())
        : otherId(oId), edgeData(eData), edgeLocation(eLocation) { }

    EdgeType getData() { return edgeData; }
    void setData(EdgeType value) { edgeData = value; }

    EdgeLocationType getEdgeLocation() { return edgeLocation; }
    void setEdgeLocation(EdgeLocationType eLoc) { edgeLocation = eLoc; }

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
class InEdge: public Edge {

public:

    InEdge(IdType sId, EdgeLocationType eLocation, EdgeType eData = EdgeType()) : Edge(sId, eLocation, eData) { }

    IdType getSourceId() { return otherId; }
    void setSourceId(IdType sId) { otherId = sId; }
};


/**
 *
 * Class of an outcoming edge.
 * 
 */
class OutEdge: public Edge {

public:

    OutEdge(IdType dId, EdgeLocationType eLocation, EdgeType eData = EdgeType()) : Edge(dId, eLocation, eData) { }
    
    IdType getDestId() { return otherId; }
    void setDestId(IdType dId) { otherId = dId; }
};


#endif /* __EDGE_HPP__ */

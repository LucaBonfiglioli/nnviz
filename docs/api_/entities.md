# 🕹️ Entities

NNViz entities serve as the **building blocks** of the neural network representation graph, consisting of nodes, edges, metadata, and other information. These object lack any kind of neural-network logic (there is no forward pass, no backprop, no weights, etc.), and are just a light-weight representation of the underlying model.

GraphData
GraphMeta
NodeModel
OpNodeModel
CollapseNodeModel
InputNodeModel
OutputNodeModel
DataSpec
TensorSpec
BuiltInSpec
UnknownSpec
ListSpec
MapSpec
DataSpecVisitor

## NNGraph

The main entity is `NNGraph`, which is a concrete type wrapping a `networkx.DiGraph` object. This object is the main representation of the neural network, and is used to store all the information about the model. Graph Nodes are represented as mappings between plain id strings and a `NodeModel`, carrying information about the node type, name, and other metadata. Graph Edges are represented as mappings between plain id strings and, optionally, a `DataSpec` object, which carries information about the data flowing through the edge. 

NNGraphs are multi-graphs, meaning that they can have multiple edges between the same pair of nodes. This is useful because some models may have a set edges that are conceptually different, but all have the same source and destination nodes. The distinction between these edges is made by an additional id string assigned when creating the edge. 

## Collapsing Operations

One of the main features of `NNGraph` is the ability to collapse multiple nodes into a single one. This is useful for simplifying the graph representation, and is done by creating a new node that represents the operation of the collapsed nodes. This new node is called a `CollapseNodeModel`, and is created and handled automatically by the `NNGraph` object.

You can currently collapse nodes by calling the collapsing methods, which act in-place on the graph. The available methods are:

- `collapse`: Collapses all nodes sharing a common path prefix into a single node.
- `collapse_multiple`: Multiple calls to `collapse` with different prefixes. This also handles the case where the prefixes are nested, and collapses the nodes in the correct order.
- `collapse_by_lambda`: Collapses all nodes based on the path prefix returned by a lambda function that takes a node id as input.
- `collapse_by_depth`: Collapses all nodes whose path prefix has a depth greater than a given value.
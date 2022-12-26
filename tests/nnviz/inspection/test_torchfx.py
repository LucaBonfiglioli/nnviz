import typing as t

import torch.fx as fx

from nnviz.inspection import torchfx as nnviz_torchfx


class TestExtendedFxGraph:
    def test_edges(self, graph_module: fx.graph_module.GraphModule):
        graph = nnviz_torchfx.ExtendedFxGraph(
            wrapped=graph_module.graph,
            qualnames={},
            callables={},
        )

        # Check that the edges are not empty.
        assert len(graph.edges) > 0

        # Check that the edges are unique.
        assert len(graph.edges) == len(set(graph.edges))

        # Check that the edges are valid.
        for arg, node in graph.edges:
            assert isinstance(arg, fx.node.Node)
            assert isinstance(node, fx.node.Node)

        # Check that the edges are in the graph.
        for arg, node in graph.edges:
            assert arg in graph.nodes
            assert node in graph.nodes

        # Check that the edges are in the correct order.
        for arg, node in graph.edges:
            assert arg in node.args or arg in node.kwargs.values()

        # Check that the edges are not self-loops.
        for arg, node in graph.edges:
            assert arg != node

    def test_nodes(self, graph_module: fx.graph_module.GraphModule):
        graph = nnviz_torchfx.ExtendedFxGraph(
            wrapped=graph_module.graph,
            qualnames={},
            callables={},
        )

        assert list(graph.nodes) == list(graph_module.graph.nodes)

    def test_qualnames(self, graph_module: fx.graph_module.GraphModule):
        qualnames: t.Dict[fx.node.Node, str] = {
            node: f"qualname_{i}" for i, node in enumerate(graph_module.graph.nodes)
        }
        graph = nnviz_torchfx.ExtendedFxGraph(
            wrapped=graph_module.graph,
            qualnames=qualnames,
            callables={},
        )

        assert graph.qualnames is qualnames

    def test_callables(self, graph_module: fx.graph_module.GraphModule):
        callables: t.Dict[fx.node.Node, t.Callable] = {
            node: lambda: None for node in graph_module.graph.nodes
        }
        graph = nnviz_torchfx.ExtendedFxGraph(
            wrapped=graph_module.graph,
            qualnames={},
            callables=callables,
        )

        assert graph.callables is callables


class TestExtendedNodePathTracer:
    def test_trace(self, graph_module: fx.graph_module.GraphModule, test_model):
        tracer = nnviz_torchfx.ExtendedNodePathTracer()
        ext_graph_module = tracer.trace(test_model)

        # Convert the nodes into plain lists
        original_nodes = list(graph_module.graph.nodes)
        extended_nodes = list(ext_graph_module.nodes)

        # Create a mapping from the original nodes to the extended nodes
        node_mapping = {}
        for original_node in original_nodes:
            for extended_node in extended_nodes:
                if original_node.name == extended_node.name:
                    node_mapping[original_node] = extended_node
                    break

        # Check that the mapping is bijective
        node_mapping_inv = {v: k for k, v in node_mapping.items()}
        assert len(node_mapping_inv) == len(node_mapping)

        # Check that the other attributes are the same
        for original_node, extended_node in node_mapping.items():
            assert original_node.op == extended_node.op
            assert original_node.target == extended_node.target
            assert original_node.name == extended_node.name
            assert original_node.type == extended_node.type

            for i, arg in enumerate(original_node.args):
                if isinstance(arg, fx.node.Node):
                    assert arg.name == extended_node.args[i].name
                else:
                    assert arg == extended_node.args[i]

            for i, (key, kwarg) in enumerate(original_node.kwargs.items()):
                if isinstance(kwarg, fx.node.Node):
                    assert kwarg.name == extended_node.kwargs[key].name
                else:
                    assert kwarg == extended_node.kwargs[key]

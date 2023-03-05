import nnviz.entities as ent


class TestNNGraph:
    # Not so great a test, I know. I am too lazy to write a better one.
    def test_to_from_data(self, collapsed_nngraph: ent.NNGraph):
        data = collapsed_nngraph.data
        nngraph = ent.NNGraph.from_data(data)
        data2 = nngraph.data
        assert data == data2

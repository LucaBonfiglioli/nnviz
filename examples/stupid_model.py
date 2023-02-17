import torch
import torch.nn as nn

import typing as t


class NormalLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class StupidLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = NormalLayer(in_features, out_features)

    def forward(self, x: t.List[torch.Tensor]) -> t.List[torch.Tensor]:
        out = self.linear(x[0])
        return [out]


class StupidModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = StupidLayer(10, 20)

    def forward(
        self,
        x: t.List[torch.Tensor],
        y: torch.Tensor,
    ) -> t.Dict[str, torch.Tensor]:
        # I hate myself for writing this
        x = x[0] + x[1] + y  # type: ignore
        x = self.layer1([x, x])
        return {"x": x[0], "y": y}


class NormalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = NormalLayer(10, 20)
        self.layer2 = NormalLayer(20, 30)
        self.layer3 = NormalLayer(30, 40)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> t.Dict[str, torch.Tensor]:
        # I hate myself for writing this
        x = x + y
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return {"x": x, "y": y}


if __name__ == "__main__":
    model = StupidModel()
    model([torch.randn(4, 10), torch.randn(4, 10)], torch.randn(4, 10))

    model = NormalModel()
    model(torch.randn(4, 10), torch.randn(4, 10))

    from torch.fx import symbolic_trace  # type: ignore

    symbolic_trace(NormalModel()).graph.print_tabular()


# Generate stupid model graph with:

# nnviz examples\stupid_model.py:StupidModel -d 1 -s -o examples\stupid_model_1_edge.pdf \
# -i "{'x': [torch.randn(3, 10), torch.randn(3, 10)], 'y': torch.rand(3, 10)}"

# This is just an example of how cursed things can get when dealing with torch.fx
# Don't ever write code like this. If you do, WIDELUCA will personally seek you out and make you regret it.

# Repent, before it's too late.

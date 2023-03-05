import pytest
import torch
import torch.nn as nn


class UntraceableModule1(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x
        else:
            return x + 10


class UntraceableModule2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.sum() < 10:
            x += 1
        return x


class UntraceableModule3(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(int(x.sum())):
            x += i
        return x


@pytest.fixture(params=[UntraceableModule1, UntraceableModule2, UntraceableModule3])
def untraceable_module(request) -> UntraceableModule1:
    return request.param()

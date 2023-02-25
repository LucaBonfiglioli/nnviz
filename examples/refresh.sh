nnviz resnet18 -d 1 -o examples/resnet18_1.pdf
nnviz resnet18 -d 2 -o examples/resnet18_2.pdf
nnviz resnet18 -d -1 -o examples/resnet18_full.pdf
nnviz resnet18 -d 1 -i default -o examples/resnet18_1_edge.pdf
nnviz resnet18 -d 2 -i default -o examples/resnet18_2_edge.pdf
nnviz resnet18 -d -1 -i default -o examples/resnet18_full_edge.pdf
nnviz examples/stupid_model.py:StupidModel -d 1 -o examples/stupid_model_1.pdf
nnviz examples/stupid_model.py:StupidModel -d 1 -o examples/stupid_model_1_edge.pdf -i "{'x': [torch.randn(3, 10), torch.randn(3, 10)], 'y': torch.rand(3, 10)}"
nnviz examples/stupid_conv.py:stupid_conv -d -1 -o examples/stupid_conv_full.pdf
nnviz examples/stupid_conv.py:stupid_conv -d -1 -o examples/stupid_conv_full_edge.pdf -i default
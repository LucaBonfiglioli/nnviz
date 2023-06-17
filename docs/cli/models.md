# 📊 Supported Models

NNViz supports model loading from the following sources:

- **Torchvision** built-in models (default)
- Models from any installed **Python package**
- Models from any local **Python file**

## Visualizing built-in torchvision models

By default, NNViz will attempt to load models from the `torchvision.models` module. You can pass the name of any model contained in `torchvision.models` and NNViz will load it for you. 

These are all valid **examples**:
```bash
nnviz resnet18
nnviz efficientnet_b0
nnviz convnext_tiny
```

## Visualizing models from installed packages

Being limited to torchvision models is not very useful, so you can also visualize models from any **installed package**, as long as they are importable. To do so, you need to pass the fully qualified name of a python **function** or **class** that returns an instance of the model you want to visualize.

Syntax: `<module>:<symbol>`

NNViz only cares that a package with the given name exists in the Python path, and that the symbol is either:
- A `nn.Module` instance
- A function that returns a `nn.Module` instance
- A `nn.Module` subclass

Examples:
```bash
nnviz package.module.submodule:my_net_instance
nnviz package.module.submodule:MyModelClass
nnviz package.module.submodule:a_function_that_returns_a_model
```

## Visualizing models from python files

And if want to visualize a model that is not in a package, you can also pass a path to a **python file** that contains the model constructor. NNViz will attempt to dynamically import the file and extract the model constructor from it.
The same rules apply as for installed packages, the only difference is that you need to pass the path to the file instead of the package name.

Syntax: `<path>:<symbol>`

```bash
nnviz path/to/file.py:my_net_instance
nnviz path/to/file.py:MyModelClass
nnviz path/to/file.py:a_function_that_returns_a_model
```

## Passing arguments

You can also pass simple built-in type **arguments** to the function, by adding `;` separated key-value pairs after the function name. Arguments are parsed with `eval`, **be careful** with what you pass.

Syntax: `<model>;<key1>=<value1>;<key2>=<value2>...`

Examples:
```bash
nnviz package.module.submodule:my_net_instance;num_classes=10;dropout=0.2
nnviz package.module.submodule:MymodelClass;width=224
nnviz package.module.submodule:a_function_that_returns_a_model;num_classes=10;dropout=0.2

nnviz path/to/file.py:my_net_instance;num_classes=10;dropout=0.2
nnviz path/to/file.py:MyModelClass;width=224
nnviz path/to/file.py:a_function_that_returns_a_model;num_classes=10;dropout=0.2
```

```{Warning}
If you feel the need to pass **complex arguments** to the function, you should probably either create a wrapper function that takes care of the arguments, or consider switching to **NNViz API** instead.
```

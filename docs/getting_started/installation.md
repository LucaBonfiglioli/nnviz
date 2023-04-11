# 🧰 Installation

**NNViz is available on PyPI** and can be installed with pip as any other python package, but there are a few requirements whose installation is not always straightforward, namely:

- [PyTorch](https://pytorch.org) (Major 2 is also supported!) - Needed to run/inspect the models, otherwise you can only use the CLI to draw static graphs. If you are installing NNViz, chances are you already have **PyTorch** installed somewhere, and in any case, you can always follow the [official installation instructions](https://pytorch.org/get-started/locally/).
- [Graphviz](https://graphviz.org/) - Needed to actually **draw** the graphs. This is a bit more tricky, as it is **OS-dependent** and requires some extra steps to install. 

## Graphviz Installation

I will now cover the installation for **Windows** and **Ubuntu**, but if you are using another OS, you can find the installation instructions on the [Graphviz website](https://graphviz.org/download/).

```{Note}
NNViz was developed mainly on a Windows environment and tested on both Windows and Ubuntu machines. Everything beside graphviz installation should be OS-independent, but currently I cannot guarantee that it will work correctly on Mac OS.
```

### Windows

Installing graphviz on Windows is a bit tricky, as it requires you to recursively build a lot of stuff in a very **plug-and-pray** process. The easiest way to skip all the hassle is to use an [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment, as they provide working binaries for graphviz. If you are not familiar with conda, I recommend you to read the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to get a better understanding of what it is and how to use it.

Once you have **conda** installed and a working python **environment**, you can install `pygraphviz` and all its dependencies with:

```bash
conda install pygraphviz
```

### Ubuntu

**Linux** users can install graphviz (and graphviz-dev) with their **package manager**, e.g. for Ubuntu:

```bash
sudo apt update
sudo apt install -y graphviz graphviz-dev
```

## NNViz Installation

Once both **PyTorch** and **Graphviz** are installed, you can install NNViz with pip:

```bash
pip install nnviz
```

## Optional Dependencies

NNViz has a few **optional dependencies** that are not required to run the package, but can be useful for debugging/development purposes. 

### Testing

If you have cloned the NNViz repository and want to **test** that everything is working correctly, you can install the `tests` extra:

```bash
pip install nnviz[tests]
```

Then, you can test the package by running, in the root folder of the repository:

```bash
pytest
```

A coverage report will be generated in the `htmlcov` folder and also printed in the terminal.

### Documentation

If you want to build the **documentation** locally, you can install the `docs` extra:

```bash
pip install nnviz[docs]
```

Then, you can build the documentation by running, in the root folder of the repository, one of the following commands, depending on your OS:

Windows:
```bash
docs.bat
```

Linux:
```bash
make docs
```

### Development

If you want to **contribute** to the project, you can install the `dev` extra:

```bash
pip install nnviz[dev]
```

This will install all the optional dev-only dependencies.

### Building the package

If you want to **build** the package locally, you can install the `build` extra:

```bash
pip install nnviz[build]
```

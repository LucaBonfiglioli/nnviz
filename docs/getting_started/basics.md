# 🍎 Basics

This section will introduce you to the basic concepts behind NNViz and network visualization in general. If you are already familiar with these concepts, you can skip this section and move on to the next one. 

If you are not familiar with deep learning, neural networks or DL frameworks as PyTorch, you may might have come across this page by mistake, as this is a tool for quickly visualizing such models, and requires you to at least have a basic understanding of these concepts.

## Neural Networks as Graphs

In the latest years, deep learning models have quickly grown in popularity and complexity, increasing the need for tools to develop them and visualize them. Modern deep learning models have evolved from simple feed-forward neural networks to complex architectures with multiple inputs and outputs, branches, loops, and more. The growing complexity of these models has led to the development of more complex and scalable tools for developing, training, and deploying them. However, understanding what is actually happening inside these models has become, to me at least, more and more difficult by just looking at the code. 

A possible solution to this "conceptualization gap" is to visualize models as hierarchical graphs, where each node represents an operation, and each edge represents the data flow between operations. This is the approach taken by NNViz (for visualization), but also by many other tools such as Torch Fx. 

Once a neural network is converted into a graph representation, it loses of course its OOP typical class structure, its flexibility and scalability, but since we are only interested in visualizing the model, this is not a problem, we only care about what operations are being performed, in what order, and what data looks like at each step, while keeping the ability to choose the level of abstraction we want to achieve. This is also the reason why most scientific papers that describe neural networks usually do so by means of figures, rather than pseudocode.

```{md-mermaid}
graph LR
    A[Input] --> B[Linear]
    B --> C[ReLU]
    C --> S
    A --> S[+]
    S --> D[Linear]
    D --> E[ReLU]
    E --> S1
    S1 --> F[Linear]
    S --> S1[+]
    F --> G[Output]
``` 

<p align="center"><i>Example of a simple neural network visualized as a graph</i></p>

In the example above, we can see a simple (and not very useful) neural network, consisting of multiple operations, among which we can find 3 linear layers, relu activations, and two residual connections. Each node represents an operation, and each edge represents the data flow between operations.

```{Note}
"Input" and "Output" are special operations that are not part of the model, but are used to represent the action of receiving and returning data from/to the outside world. 
```

Try to implement a network like that in PyTorch, and you will quickly realize that the code, despite being very simple, does not convey the information as clearly and concisely as the graph representation does. If you think otherwise, you should immediately stop what you are doing and consider seeking professional help.

## Module Hierarchies

What about the hierarchical structure of the model? Considering the example in the previous section, chances are that, in your code, you have a class named `LinearBlock` (or something like that) that inherits from `nn.Module` and wraps the underlying `nn.Linear`, the activation and the residual connection. The fact that you have this class is not a coincidence, it is a (correct) design choice that:
- Gives a name to a group of operations
- Enables you to reuse the same block in multiple places
- Creates an abstraction layer between the low-level block and the high-level model architecture
  
But all these benefits are lost when you consider the previous diagram, where all the details of the block are shown. This is where the hierarchical structure of the graph comes into play. Consider the two following examples:

```{md-mermaid}
graph LR
    subgraph LinearBlock
        B[Linear]
        B --> C[ReLU]
        C --> S
    end
    A[Input] --> S[+]
    A --> B
    subgraph LinearBlock2[LinearBlock]
        E[Linear]
        E --> F[ReLU]
        F --> S1
        S --> S1
    end
    S --> E
    S1[+] --> G[Linear]
    G --> H[Output]
``` 
<p align="center"><i>Example of a simple neural network visualized as a graph, with a hierarchical structure</i></p>

```{md-mermaid}
graph LR
    A[Input] --> B[LinearBlock]
    B --> C[LinearBlock]
    C --> D[Linear]
    D --> E[Output]
```

<p align="center"><i>The same model visualized at a higher abstraction level</i></p>

In the first example, we can see that `LinearBlock`s are represented as subgraphs, and that the model is composed of multiple `LinearBlock`s, preserving the hierarchical structure given by OOP design. These subgraphs can be collapsed and expanded, allowing us to choose a different level of abstraction. In the second example, we can see that the model is represented as a single graph, where the `LinearBlock`s are represented as a single node, abstracting away the details of the block. 

## Static and Dynamic Models

```{Note}
These terms are made up by me to clarify an important distinction, and are not part of any standard terminology. I will therefore write them in italics, to make it clear that they are not official terms. 
```

Before we move on, we need to make an important distinction between ***static*** and ***dynamic*** models. In *static* models, every operation is known at the time of model definition, and it will never change. In *dynamic* models, on the other hand, the exact operations that will be performed are determined at execution time, and depend on the input data. 

```{Note}
The mere presence of conditional statements, loops, recursions or any other control flow mechanism does not necessarily mean that the model is *dynamic*. You can have as many conditional statements as you want, and still have a *static* model, as long as they do not depend on the input data. 
```

Let's clarify this distinction with a PyTorch example. Consider the following model:

```python
class MyModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(n)])
    
    def forward(self, x):
        for i in range(self.n):
            x = self.layers[i](x)
        return x
```

This model contains a `ModuleList` of `Linear` layers, and a `for` loop that iterates over the layers and applies them to the input data. Despite the fact that this model contains a `for` loop, it is still *static*, because **once the model is defined**, the number of layers is **fixed**, and it will never change. If you pass a random tensor to the model and record the operations performed on it, and you will see that the same operations are performed every time, regardless of the input data.

Consider now the following model:

```python
class MyModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(n)])
    
    def forward(self, x):
        for i in range(self.n):
            if x.mean() > 0:
                x = self.layers[i](x)
        return x
```

This model is *dynamic*, because the number of layers that will be applied to the input data depends on the input data itself. If you pass a random tensor to the model and record the operations performed on it, you will see that the number of layers applied to the input data **varies** from one execution to another, **depending on the input data**.

NNViz can be used to visualize only *static* models, and **it will not work** with *dynamic* ones. This is because, in order to visualize a model, NNViz needs to know the exact operations that will be performed on the input data, and this information is simply not available in *dynamic* models. 

There are some possible **solutions** to this problem:

1. Design and implement an abstraction that can be used to represent *dynamic* models as graphs. This is not an easy task, and it is not something that I am planning to do in the near future. If you are interested in this topic and have some ideas, feel free to open an issue on GitHub and discuss it with me.

2. Create an inspection mechanism that detects *dynamic* parts of the model, and replaces them with a blackbox placeholder, allowing the visualization of the rest of the model. This is a much easier task, and much more likely to be implemented in the future.

3. Use NNViz to visualize only the *static* parts of the model.

## What does NNViz do?

## Visualizing a Model
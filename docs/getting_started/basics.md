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

## What does NNViz do?

## Visualizing a Model
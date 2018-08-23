# RNN dynamical analysis
## I. Task description
### 1. Sequential MNIST classification
The 784-pixel images are reshaped into 28 dimensions by 28 time steps. There are 10 classes for digit 0-9, respectively.
### 2. DNA classification
Splice-junction Gene Sequences Data Set 

https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)

Each DNA sequence has 60 base pairs. Therefore the input has 4 dimensions (A/T/G/C) and 60 time steps.

> Splice junctions are points on a DNA sequence at which "superfluous" DNA is removed during the process of protein creation in higher organisms. The problem posed in this dataset is to recognize, given a sequence of DNA, the boundaries between exons (the parts of the DNA sequence retained after splicing) and introns (the parts of the DNA sequence that are spliced out). This problem consists of two subtasks: recognizing exon/intron boundaries (referred to as EI sites), and recognizing intron/exon boundaries (IE sites). (In the biological community, IE borders are referred to as "acceptors" while EI borders are referred to as ``donors''.) 

There are 3 classes in total: EI/IE and N. But this data set can also be reduced and used for binary classification task.

## II. Experiment
### RNN setup

As of now, the RNNs are built with Keras, and only LSTM and GRU were tested.

A RNN model contains a recurrent layer and a dense layer.

### Performance evaluation

1. Sequential mnist

N of units | Accuracy
----------- | ------------
8 | 0.8842
16 | 0.9521
32 | 0.9722
64| 0.9834
128| 0.9851
256 | 0.9878

2. DNA classification

N of units | Accuracy
----------- | -----------
4 | 0.8730
6 | 0.8925
8 | 0.9023
10 | 0.9349
12 | 0.9218
16 | 0.9283
32 | 0.9349
64 | 0.9446
128 | 0.9381

### Observations on dynamics
(data not shown here)
* The predictions made by RNN model change smoothly with time steps. The output values are usually close to the target value even though t!=sequence_length 
* When a sequence ends, keep feeding in zero input will let the output converge to some fixed points. This procedure can take very long time (10x the time steps of the original task). These fixed points are often not unique, even though the output space is low-dimensional. **The main goal of this project starts from here.**

## III. TODOs
### Hypothesis
* Bijection exists between the fixed points in hidden state space and in the output space. 
* Correlation exists between the input data and the fixed points.

### Visualization
* Hidden state space
* Trajectories of data sequence in hidden state space
* Perturbations to the trajectories

## IV. Previous related work
The previous research on robot arm controllers are dynamical analysis on autonommous dynamical systems. The RNN models in this project are non-autonomous dynamical systems, which have more complicated dynamics. 

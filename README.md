# Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models

**Authors:** Frederiek Wesel, Kim Batselier

## Abstract

In the context of kernel machines, polynomial and Fourier features are commonly used to provide a nonlinear extension to linear models by mapping the data to a higher-dimensional space. Unless one considers the dual formulation of the learning problem, which renders exact large-scale learning unfeasible, the exponential increase of model parameters in the dimensionality of the data caused by their tensor-product structure prohibits to tackle high-dimensional problems. One of the possible approaches to circumvent this exponential scaling is to exploit the tensor structure present in the features by constraining the model weights to be an underparametrized tensor network. In this paper we quantize, i.e. further tensorize, polynomial and Fourier features. Based on this feature quantization we propose to quantize the associated model weights, yielding quantized models. We show that, for the same number of model parameters, the resulting quantized models have a higher bound on the VC-dimension as opposed to their non-quantized counterparts, at no additional computational cost while learning from identical features. We verify experimentally how this additional tensorization regularizes the learning problem by prioritizing the most salient features in the data and how it provides models with increased generalization capabilities. We finally benchmark our approach on large regression task, achieving state-of-the-art results on a laptop computer.

## Data

All datasets can be downloaded from the webpage of Andrew Wilson [here](https://people.orie.cornell.edu/andrew/pattern/#Data).

Instructions to download the airline dataset will follow.

## Reproducing Results

To replicate the figures and results in the paper, it is necessary to run the corresponding script provided in the repository.

## Paper

The paper associated with this work can be found [here](https://proceedings.mlr.press/v238/wesel24a.html).

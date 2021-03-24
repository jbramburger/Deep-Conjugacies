# **Deep Learning of Conjugate Mappings**

This repository contains scripts and notebooks to reproduce the data and figures from Deep Learning of Conjugate Mappings by Jason J. Bramburger, Steven L. Brunton, and J. Nathan Kutz (2021).

## **Paper Abstract**
Despite many of the most common chaotic dynamical systems being continuous in time, it is through discrete time mappings that much of the understanding of chaos is formed. Henri Poincare first made this connection by tracking consecutive iterations of the continuous flow with a lower-dimensional, transverse subspace. The mapping that iterates the dynamics through consecutive intersections of the flow with the subspace is now referred to as a Poincare map, and it is the primary method available for interpreting and classifying chaotic dynamics. Unfortunately, in all but the simplest systems, an explicit form for such a mapping remains outstanding. This work proposes a method for obtaining explicit Poincare mappings by using deep learning to construct an invertible coordinate transformation into a conjugate representation where the dynamics are governed by a relatively simple chaotic mapping. The invertible change of variable is based on an autoencoder, which allows for dimensionality reduction, and has the advantage of classifying chaotic systems using the equivalence relation of topological conjugacies. Indeed, the enforcement of topological conjugacies is the critical neural network regularization for learning the coordinate and dynamics pairing. We provide expository applications of the method to low-dimensional systems such as the Rossler and Lorenz systems, while also demonstrating the utility of the method on infinite-dimensional systems, such as the Kuramoto-Sivashinsky equation.

## **Network Architecture**
Our goal with this project is to find an explicit Poincare mapping which fits data gathered in a suitable Poincare section from a continuous-time dynamical system. We employ a feed-forward neural network architecture to circumvent discovering the Poincare mapping directly. Precisely, the network exploits the concept of a topological conjugacy and discovers an invertible change of variables along with a 'simple' mapping that governs iterates of the transformed observations.  
 

## **Repository Contents**


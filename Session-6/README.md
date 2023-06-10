
Assignment 6 – ERA 1 
==========================

- Developer(s)
     - Nihar Kanungo  (nihar.kanungo@gmail.com)


Requirement
==========

Part-1
----------

Most of us know how a neural network model optimizes itself in multiple iterations by reducing the loss and thereby increasing the accuracy of correct prediction. It’s very simple, isn’t it? Probably couple of lines of code. However, have we ever thought how difficult our life would have been without these? Why only we possibly everyone in this world who are associated with data mining, data analysis, ML etc.
But how do we ensure that we understand the pain and hence respect those great minds who developed this for us? Perhaps that’s the logic behind getting this as our assignment. So here we go 

![](images/simple_perceptron_model.png)

The above diagram is simple 
We need to write a neural network which includes 

1.	2 input value (Yes Just two, not the FP16 or FP32 representation)
2.	One Hidden layer of size 2 
3.	Output layer
4.	2 Output values 
Optimize it over some epoch and see how the loss decreases.
Perform the same experiment for different Learning Rates 
Compare and see how each network learns and visualize 

Part-2
---------
Train a Neural Network on MNIST dataset.
Simple isn’t it ?
.
.
.
.
But here is the twist.

Constraints
============
1.	The Parameters must be with in 12k – 18k 
2.	We must run the model for exactly 19 epochs 
3.	Must use Batch Normalization, FC Layer and GAP 
4.	Dropout value must be 0.069 only 

What to achieve 
================
   -  '>=' 99.4% accuracy with less than 20 epochs 

What’s the right approach
===========================
The Training must be smooth and the validation accuracy should be 99.4 % over multiple epochs. If we plot the graph then it must be a smooth graph  

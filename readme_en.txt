NNSys 0.1 software system is meant to simulate and train feedforward multilayer neural networks with sigmoid activation functions (multilayer perceptrons).

Calculations within layers are parallelized between neurons by means of OpenMP technology. Five learning algorithms based on the gradient descent method are implemented (there are classical backward propagation of errors and its modifications). Three stopping criterion can be used: 1) obtaining the required minimum of mean-square training error; 2) obtaining the required minimum of training error's gradient norm; 3) growth of the generalization error. Last criterion is named early stopping criterion. Its applying allows to avert overfitting of the trained neural network.

In future the NNSys will be improved in the following way: 1) genetic algorithm will be added to the list of implemented learning algorithms; 2) pre-training algorithm based on deep belief networks will be implemented for preparing to final training by backprop; 3) OpenCL technology instead of OpenMP will be used for parallel computing.

Main features of NNSys 0.1 software system are described below.

1. Create multilayer perceptrons with any number of layers and neurons.

2. Use two kinds of activation function: rational sigmoid and linear function.

3. Speed up the neural computing on multicore CPU by means of using OpenMP.

4. Train neural networks by one of five algorithms: online backprop, Incremental Delta-bar-Delta, batch backprop, Rprop (resilient backpropagation), and conjugate gradients algorithm.

5. Use combination of three stopping criterion: 1) through obtaining the required minimum of mean-square training error; 2) through obtaining the required minimum of training error's gradient norm; 3) in view of growth of the generalization error.

6. Calculate either classification error or regression error (besides traditional calculation of mean-square error) during testing of multilayer perceptrons.

7. Convert training and testing data from well-known CSV format to the NNSys own  format of train sets (and conversely).

NNSys 0.1 software system is propagated in compliance with version 3 of GNU General Public License. User guide (in Russian) and files with source code are included in the supply. User guide in English will be written in the recent future.

For compilation of this software system it is necessary to use 1) any compiler which supports ISO/IEC 14882:1998 standard; 2) installed Qt library of either version 4.8 or any later version.

Copyright Ivan Yu. Bondarenko, 2014
email: bond005@yandex.ru
skype: i_yu_bondarenko
web:   http://ua.linkedin.com/pub/ivan-bondarenko/3/785/632

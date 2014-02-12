Software system NNSys 0.1 is meant to simulate and train feedforward multilayer neural networks with sigmoid activation functions (multilayer perceptrons).

Calculations within layers are parallelized between neurons by means of OpenMP technology. Five learning algorithms based on the gradient descent method are implemented (there are classical backward propagation of errors and its modifications). Three stopping criterion can be used: 1) obtaining the required minimum of mean-square training error; 2) obtaining the required minimum of training error's gradient norm; 3) growth of the generalization error. Last criterion is named early stopping criterion. Its applying allows to avert overfitting of the trained neural network.

Software system NNSys 0.1 is propagated in compliance with version 3 of GNU General Public License. User guide (in Russian) and files with source code are included in the supply. User guide in English will be written in the recent future.

For compilation of this software system it is necessary to use 1) any compiler which supports ISO/IEC 14882:1998 standard; 2) installed Qt library of either version 4.8 or any later version.

Copyright Ivan Yu. Bondarenko, 2014
email: bond005@yandex.ru
skype: i_yu_bondarenko
web:   http://ua.linkedin.com/pub/ivan-bondarenko/3/785/632

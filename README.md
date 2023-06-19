# PIF
Unlearning via Partial Influence Function.
To be updated...

# Experiments in Sec. 2.2

# Training the models
```
run/recursive_model_train.sh LeNet cross_entropy CIFAR10
run/recursive_model_train.sh LeNet mse CIFAR10
run/recursive_model_train.sh FCN cross_entropy MNIST
run/recursive_model_train.sh FCN mse MNIST
```
# Acknowledgements

The art of programming is reproduction.
Here is the list of repos I refered.
I deeply appreciate for the authors of the repos.

* Models: [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* Lanczos method:[noahgolmant/pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings)

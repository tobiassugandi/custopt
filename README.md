Custopt includes a Pytorch implementation of Nys-LBFGS, a modified LBFGS optimization algorithm--using the randomized Nystrom approximation as the initial guess of the inverse Hessian approximation.

### Motivation 
Enhance the quality of LBFGS's inverse Hessian approximation to better address ill-conditioned problems. Initially devised to mitigate the training issue in physics informed neural operators.

### Variants 
- Full batch
- Full overlap: stable implementation of LBFGS with mini-batching

### Inspiration
This implementation builds on the following works:
- https://github.com/hjmshi/PyTorch-LBFGS
- https://github.com/pratikrathore8/opt_for_pinns

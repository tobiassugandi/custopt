'''
Adapted from the implementation of Rathore et al. (2024) (https://github.com/pratikrathore8/opt_for_pinns), 
which in turn is adapted from:
  Original paper: PyHessian: Neural Networks Through the Lens of the Hessian (https://arxiv.org/abs/1912.07145)
  Original authors implementation: https://github.com/amirgholami/PyHessian
'''

import torch
import numpy as np

### utility functions ###

"""
Compute the inner product of two lists of tensors xs, ys.
"""
def list_dot_product(xs, ys):
  return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

"""
Compute the updated list of tensors (params) in the corresponding list of direction tensors (update) with given step size (alpha). 
"""
def group_add(params, update, alpha=1):
  for i, p in enumerate(params):
    p.data.add_(update[i], alpha=alpha)
  return params

"""
Normalize a vector (represented as a list of tensors). . 
"""
def normalization(v):
  s = list_dot_product(v, v)
  s = s ** 0.5
  s = s.cpu().item()
  v = [vi / (s + 1e-6) for vi in v]
  return v

"""
Orthonormalize vector w to a list of vectors (v_list). 
"""
def orthonormalization(w, v_list):
  for v in v_list:
    w = group_add(w, v, alpha=-list_dot_product(w, v))
  return normalization(w)

"""
Compute the HVP where gradsH is the gradient at the current point, params is the corresponding variables, and v is the vector. 
"""
def hessian_vector_product(gradsH, params, v):
  hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True)
  return hv

### hessian class ###

class Hessian():
  """
  Class for computing spectral density of (L-BFGS) pre-conditioned Hessian. 

  - model: instance of NN
  - grad_tuple: tuple of gradients of the loss w.r.t. the model parameters
  - device: string indicating which device to use
  """
  def __init__(self, model, grad_tuple, device='cuda'):
    self.model = model.eval()
    self.device = device
    self.params = [param for param in self.model.parameters() if param.requires_grad]
    self.gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]


  """
  Function for performing spectral density computation using Stochastic Lanczos Quadrature. 

  INPUT: 
  - num_iter: number of Lanczos iterations
  - num_run: number of Lanczos quadrature repetitions (number of samples for the expectation)
  OUTPUT: 
  - eigen_list_full: list eigenvalues for each run
  - weight_list_full: list of corresponding densities for each run
  """
  def density(self, num_iter=100, num_run=1):
    eigen_list_full = []
    weight_list_full = []

    for k in range(num_run):
      # generate Rademacher random vector
      v = [2 * torch.randint_like(p, high=2, device=self.device) - 1 for p in self.params]
      v = normalization(v)

      # Lanczos initlization
      v_list = [v]
      # w_list = []
      alpha_list = []
      beta_list = []

      # run Lanczos
      for i in range(num_iter):
        self.model.zero_grad()
        w_prime = [torch.zeros(p.size()).to(self.device) for p in self.params]
        if i == 0:
          w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = list_dot_product(w_prime, v)
          alpha_list.append(alpha.cpu().item())
          w = group_add(w_prime, v, alpha=-alpha)
          # w_list.append(w)
        else:
          beta = torch.sqrt(list_dot_product(w, w))
          beta_list.append(beta.cpu().item())
          if beta_list[-1] != 0.:
            v = orthonormalization(w, v_list)
            v_list.append(v)
          else:
            w = [torch.randn(p.size()).to(self.device) for p in self.params]
            v = orthonormalization(w, v_list)
            v_list.append(v)
          w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = list_dot_product(w_prime, v)
          alpha_list.append(alpha.cpu().item())
          w_tmp = group_add(w_prime, v, alpha=-alpha)
          w = group_add(w_tmp, v_list[-2], alpha=-beta)

      # piece together tridiagonal matrix
      T = torch.zeros(num_iter, num_iter).to(self.device)
      for i in range(len(alpha_list)):
        T[i, i] = alpha_list[i]
        if i < len(alpha_list) - 1:
          T[i + 1, i] = beta_list[i]
          T[i, i + 1] = beta_list[i]

      eigenvalues, eigenvectors = torch.linalg.eig(T)

      eigen_list = eigenvalues.real
      weight_list = torch.pow(eigenvectors[0,:], 2)
      eigen_list_full.append(list(eigen_list.cpu().numpy()))
      weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigen_list_full, weight_list_full

  """
  Function for finding eigenvalues and eigenvectors using power iteration. 

  INPUT: 
  - max_num_iter: maximum number of iterations for each eigenvalue
  - top_n: number of top n eigenvalues to discover
  OUTPUT: 
  - eigenvalues: eigenvalues
  - eigenvectors: corresponding eigenvectors
  - iter_used: number of iterations until convergence for each eigenvalue
  """
  def eigenvalues(self, max_num_iter=100, tol=1e-3, top_n=1):
    assert top_n >= 1

    device = self.device

    eigenvalues = []
    eigenvectors = []
    iter_used = np.zeros(top_n)

    computed_dim = 0

    while computed_dim < top_n:
        eigenvalue = None
        v = [torch.randn(p.size()).to(device) for p in self.params]  # generate random vector
        v = normalization(v)  # normalize the vector

        for i in range(max_num_iter):
            iter_used[computed_dim] += 1
            v = orthonormalization(v, eigenvectors)
            self.model.zero_grad()

            Hv = hessian_vector_product(self.gradsH, self.params, v)
            tmp_eigenvalue = list_dot_product(Hv, v).cpu().item()

            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1

    return eigenvalues, eigenvectors, iter_used
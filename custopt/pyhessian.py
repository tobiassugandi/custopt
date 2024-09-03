'''
Original paper: PyHessian: Neural Networks Through the Lens of the Hessian (https://arxiv.org/abs/1912.07145)
Original authors implementation: https://github.com/amirgholami/PyHessian

Code is adapted to work with PINNs. 
'''

import torch
import numpy as np

### utility functions ###

"""
Compute the inner product of two lists of tensors xs, ys.
"""
def group_product(xs, ys):
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
  s = group_product(v, v)
  s = s ** 0.5
  s = s.cpu().item()
  v = [vi / (s + 1e-6) for vi in v]
  return v

"""
Orthonormalize vector w to a list of vectors (v_list). 
"""
def orthonormalization(w, v_list):
  for v in v_list:
    w = group_add(w, v, alpha=-group_product(w, v))
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

  - model: instance of PINN
  - pred_func: prediction function
  - loss_func: loss function
  - data: tuple of spatial and temporal inputs
  - loss_comp: string indicating which component of the loss to use (will use total loss for any value other than "res", "bc", or "ic")
  - device: string indicating which CUDA device to use (where both model and data reside)
  """
  def __init__(self, model, grad_tuple, device='cuda'):
    self.model = model.eval()
    self.device = device
    self.params = [param for param in self.model.parameters() if param.requires_grad]
    self.gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]



  # def __init__(self, model, data, loss_comp=None, device='cuda'):
  #   self.model = model.eval()
  #   # self.pred_func = pred_func
  #   # self.loss_func = loss_func
  #   # self.x, self.t = data
  #   self.device = device


  #   def compute_fourier_derivative(u, s=None, domain_length = 2 ):
  #     if s is None:
  #         s = u.shape[-1]
          
  #     # pad = 0
  #     # padded = torch.nn.functional.pad(u, (pad,pad), mode='circular')

  #     # Compute Fourier transform
  #     output_fft = torch.fft.fft(u, dim=-1)

  #     # Create the modes
  #     k_max = s//2
  #     k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
  #                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,s)
  #     k =k * 2j * torch.pi / domain_length

  #     # Multiply with modes to compute derivative in the frequency domain
  #     d_fft = k * output_fft
  #     d2_fft = k ** 2  * output_fft 

  #     # print(d_fft.shape, d2_fft.shape)    
  #     du, d2u = torch.fft.irfft(d_fft, n=s, dim=-1), torch.fft.irfft(d2_fft, n=s, dim=-1)
  #     du, d2u = du, d2u
  #     # print(du.shape, d2u.shape)

  #     return du, d2u
    
  #   l = torch.nn.L1Loss()
  #   self.model.zero_grad()

  #   input_batch, output_batch, out_data_std = data
  #   output_pred_batch   = model(input_batch) 
  #   loss_data           = l(output_pred_batch, output_batch) \
  #                         + torch.nn.MSELoss()(output_pred_batch * 0, torch.zeros_like(output_pred_batch))  # second term ensures all model parameters are in the graph

  #   out_ori             = output_pred_batch * out_data_std
  #   du, d2u             = compute_fourier_derivative(out_ori)
  #   declude             = 1
  #   loss_pde            = l(-d2u[:, 0, declude:-declude], input_batch[:, 1, declude:-declude])

  #   loss                = loss_data + loss_pde

  #   grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        
  #   self.params = [param for param in self.model.parameters() if param.requires_grad]
  #   self.gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]

  """
  Function for performing spectral density computation. 

  INPUT: 
  - num_iter: number of iterations for Lanczos
  - num_run: number of runs
  OUTPUT: 
  - eigen_list_full: list eigenvalues for each run
  - weight_list_full: list of corresponding densities for each run
  """
  def density(self, num_iter=100, num_run=1):
    eigen_list_full = []
    weight_list_full = []

    for k in range(num_run):
      # generate Rademacher random variables
      v = [2 * torch.randint_like(p, high=2, device=self.device) - 1 for p in self.params]
      v = normalization(v)

      # Lanczos initlization
      v_list = [v]
      w_list = []
      alpha_list = []
      beta_list = []

      # run Lanczos
      for i in range(num_iter):
        self.model.zero_grad()
        w_prime = [torch.zeros(p.size()).to(self.device) for p in self.params]
        if i == 0:
          w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = group_product(w_prime, v)
          alpha_list.append(alpha.cpu().item())
          w = group_add(w_prime, v, alpha=-alpha)
          w_list.append(w)
        else:
          beta = torch.sqrt(group_product(w, w))
          beta_list.append(beta.cpu().item())
          if beta_list[-1] != 0.:
            v = orthonormalization(w, v_list)
            v_list.append(v)
          else:
            w = [torch.randn(p.size()).to(self.device) for p in self.params]
            v = orthonormalization(w, v_list)
            v_list.append(v)
          w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = group_product(w_prime, v)
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
            tmp_eigenvalue = group_product(Hv, v).cpu().item()

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
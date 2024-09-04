'''
Adapted from the implementation of Rathore et al. (2024) (https://github.com/pratikrathore8/opt_for_pinns)
'''

import torch
from .nysopt import _apply_nys_inv

class Precond_hessian():
  """
  Class for computing spectral density of ((Nys-)L-BFGS) pre-conditioned Hessian. 

  - optimizer: L-BFGS instance used to optimize the model
  - model: instance of PINN
  - grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True
  - device: string indicating which device to use (where both model and data reside)
  """
  def __init__(self, optimizer, model, grad_tuple, device='cuda'):
    self.model      = model.eval() 
    self.device     = device
    # self.optimizer  = optimizer

    self.params = [param for param in self.model.parameters() if param.requires_grad]
    self.param_length = sum( 2 * p.numel() if torch.is_complex(p) else p.numel() for p in self.params ) # optimizer._numel_cache
    # grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
    self.gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]

    # Extract L-BFGS variables
    try: # pytorch's L-BFGS
      state = optimizer.state_dict()["state"][0]
      history_size = len(state['old_dirs'])
      y_hist    = state['old_dirs']
      s_hist    = state['old_stps']
      rho_hist  = state['ro']
      gamma     = state['H_diag']
    except: # hjmshi L-BFGS
      state       = optimizer.state["global_state"]
      history_size = len(state['old_dirs'])
      y_hist      = [y.to(device) for y in state['old_dirs'] ]
      s_hist      = [s.to(device) for s in state['old_stps'] ]
      rho_hist    = [rho.to(device) for rho in state['rho'] ]
      gamma       = state['H_diag'].to(device)
      
      self.H0     = state.get("H0")
      self.U      = state.get("U")
      self.S      = state.get("S")
      self.nys_mu = state.get("nys_mu")

    self.history_size   = history_size
    self.y_hist         = y_hist
    self.rho_hist       = rho_hist
    self.gamma          = gamma


    # Compute: The factorization tilde_H_k tilde_H_k^T of the LBFGS inverse hessian
    tilde_v_vecs = [None] * history_size
    tilde_s_vecs = [None] * history_size

    tilde_v_vecs[-1] = s_hist[-1]
    tilde_s_vecs[-1] = rho_hist[-1].sqrt() * s_hist[-1]

    for i in range(history_size-2, -1, -1):
      # compute tilde_s_i
      tilde_s_updates = [y.dot(s_hist[i]) * rho * tilde_v for rho, y, tilde_v in zip(rho_hist[i+1:], y_hist[i+1:], tilde_v_vecs[i+1:])]
      tilde_s_vecs[i] = rho_hist[i].sqrt() * (s_hist[i] - torch.stack(tilde_s_updates, 0).sum(0))
      # compute tilde_v_i
      tilde_v_terms = [rho * y.dot(s_hist[i]) * tilde_v for rho, y, tilde_v in zip(rho_hist[i+1:], y_hist[i+1:], tilde_v_vecs[i+1:])]
      tilde_v_vecs[i] = s_hist[i] - torch.stack(tilde_v_terms, 0).sum(0)

    self.tilde_v_vecs = tilde_v_vecs
    self.tilde_s_vecs = tilde_s_vecs





  """
  Function for computing matrix vector product of matrix 
    (P^-1/2 Hessian P^-1/2) = (tilde_H_k.T Hessian tilde_H_k) and vector v. 

  INPUT: 
  - v: tensor of size (num of model params + history size of L-BFGS)
  OUTPUT: 
  - mv: tensor of size (num of model params + history size of L-BFGS)
  """
  def matrix_vector_product(self, v): 
    # step 1: compute mvp tilde_H_k @ v
    # compute v_prime
    v1 = v[:-self.history_size]
    v2 = v[-self.history_size:]
    if not self.H0 == "Nys":
      v1 = self.gamma.sqrt() * v1
    else:
      v1 = _apply_nys_inv(v1, self.U, self.S, self.nys_mu, 
                                    pow=-0.5)
    v_prime = [rho * y.dot(v1) for rho, y in zip(self.rho_hist, self.y_hist)]
    v_prime = v1 - torch.stack([v_i * tilde_v for v_i, tilde_v in zip(v_prime, self.tilde_v_vecs)], 0).sum(0)
    v_prime = v_prime + torch.stack([v_i * tilde_s for v_i, tilde_s in zip(v2, self.tilde_s_vecs)], 0).sum(0)
    
    # step 2: compute Hv_prime using autograd
    # convert tensor to a list of tensors matching model parameters
    v_prime_list = []
    offset = 0
    for p in self.params: 
      numel = p.numel()
      v_prime_list.append( v_prime[offset : offset + numel].reshape( p.size() ) )
      offset += numel
    hv_prime = torch.autograd.grad(self.gradsH, self.params, grad_outputs=v_prime_list, only_inputs=True, retain_graph=True)
    # flatten result
    views = []
    for p in hv_prime:
      views.append(p.reshape(-1))
    hv_prime = torch.cat(views, 0)
    
    # step 3: compute mvp tilde_H_k^T @ Hv_prime
    v1 = [tilde_v.dot(hv_prime) for tilde_v in self.tilde_v_vecs]
    v1 = torch.stack([v_i * rho * y for rho, y, v_i in zip(self.rho_hist, self.y_hist, v1)], 0).sum(0)
    if not self.H0 == "Nys":
      v1 = self.gamma.sqrt() * (hv_prime - v1)
    else:
      v1 = _apply_nys_inv((hv_prime - v1), self.U, self.S, self.nys_mu, 
                               pow=-0.5)
    v2 = torch.tensor([tilde_s.dot(hv_prime) for tilde_s in self.tilde_s_vecs], device=self.device)
    
    return torch.cat([v1, v2], 0)

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
      # generate Rademacher random vector
      v = (2 * torch.randint(high=2, size=(self.param_length+self.history_size,), device=self.device, dtype=torch.get_default_dtype()) - 1)
      v = self._normalization(v)

      # Lanczos initlization
      v_list = [v]
      # w_list = []
      alpha_list = []
      beta_list = []

      # run Lanczos
      for i in range(num_iter):
        self.model.zero_grad()
        w_prime = torch.zeros(self.param_length+self.history_size).to(self.device)
        if i == 0:
          w_prime = self.matrix_vector_product(v)
          alpha = w_prime.dot(v)
          alpha_list.append(alpha.cpu().item())
          w = w_prime - alpha * v
          # w_list.append(w)
        else:
          beta = torch.sqrt(w.dot(w))
          beta_list.append(beta.cpu().item())
          if beta_list[-1] != 0.:
            v = self._orthonormalization(w, v_list)
            v_list.append(v)
          else:
            w = torch.randn(self.param_length+self.history_size).to(self.device)
            v = self._orthonormalization(w, v_list)
            v_list.append(v)
          w_prime = self.matrix_vector_product(v)
          alpha = w_prime.dot(v)
          alpha_list.append(alpha.cpu().item())
          w_tmp = w_prime - alpha * v
          w = w_tmp - beta * v_list[-2]

      # piece together tridiagonal matrix
      T = torch.zeros(num_iter, num_iter).to(self.device)
      for i in range(len(alpha_list)):
        T[i, i] = alpha_list[i]
        if i < len(alpha_list) - 1:
          T[i + 1, i] = beta_list[i]
          T[i, i + 1] = beta_list[i]

      eigenvalues, eigenvectors = torch.linalg.eig(T)

      eigen_list = eigenvalues.real
      weight_list = torch.pow(eigenvectors[0,:], 2) # only stores the square of first component of eigenvectors
      eigen_list_full.append(list(eigen_list.cpu().numpy()))
      weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigen_list_full, weight_list_full

  """
  Helper function for normalizing a given vector. 
  """
  def _normalization(self, w): 
    return w / (w.norm() + 1e-6)

  """
  Helper function for orthonormalize given vector w with respect to a list of vectors (Gram–Schmidt + normalization). 
  """
  def _orthonormalization(self, w, v_list): 
    for v in v_list:
      w = w - w.dot(v) * v
    return self._normalization(w)
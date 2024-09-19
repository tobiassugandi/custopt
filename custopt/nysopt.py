"""
A collection of classes and helper functions for optimizers that utilize Nyström approximation.

This implementation is adapted from the code of the following paper:
    `Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective.
    Preprint, 2024. <https://arxiv.org/abs/2402.01868>`
    (https://github.com/pratikrathore8/opt_for_pinns)
"""

import torch
from torch.optim import Optimizer
from torch.func import vmap
from functools import reduce

from .line_search import _armijo


@torch.no_grad()
def _apply_nys_inv(x, U, S, mu, pow = -1.0):
    """Applies the inverse of the Nystrom approximation of the (regularized) Hessian to a vector."""
    S_mu_inv    = (S + mu) ** pow # eigenvalue inverse

    # U (S + mu)^pow U^T x + (I - U * U^T ) x * mu^pow
    z = U.T @ x
    if mu > 1e-6:
        z = (U @ (S_mu_inv * z)) + (x - U @ z) * mu ** pow 
    else:
        z = (U @ (S_mu_inv * z))
    return z

@torch.no_grad()
def _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, x):
    """Applies the inverse of the Nystrom preconditioner of the Hessian to a vector."""
    z = U.T @ x
    z = (lambd_r + mu) * (U @ (S_mu_inv * z)) + (x - U @ z)
    return z

def _nystrom_pcg(hess, b, x, mu, U, S, r, tol, max_iters):
    """Solves a positive-definite linear system using NyströmPCG.

    `Frangella et al. Randomized Nyström Preconditioning. 
    SIAM Journal on Matrix Analysis and Applications, 2023.
    <https://epubs.siam.org/doi/10.1137/21M1466244>`"""
    lambd_r = S[r - 1]
    S_mu_inv = (S + mu) ** (-1)

    resid = b - (hess(x) + mu * x)
    with torch.no_grad():
        z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
        p = z.clone()

    i = 0

    while torch.norm(resid) > tol and i < max_iters:
        v = hess(p) + mu * p
        with torch.no_grad():
            alpha = torch.vdot(resid, z) / torch.vdot(p, v)
            x += alpha * p

            rTz = torch.vdot(resid, z)
            resid -= alpha * v
            z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
            beta = torch.vdot(resid, z) / rTz

            p = z + beta * p

        i += 1

    if torch.norm(resid) > tol:
        print(f"Warning: PCG did not converge to tolerance. Tolerance was {tol} but norm of residual is {torch.norm(resid)}")

    return x

def _adaptive_nys_hess_approx(grad_tuple, _params_list, chunk_size, verbose, rank0 = 15, adaptive = True, mu = 1e-2, max_rank = 100, rank_mult = 2):
    """Adaptive rank selection for the Nystrom approximation of the Hessian. 
    The rank doubles until the ratio of the k-th eigenvalue to the matrix regularizer mu is less than 10 (Frangella et al., 23). 

    Args:
        grad_tuple (tuple): tuple of Tensors containing the gradients of the loss w.r.t. the parameters. 
            This tuple can be obtained by calling torch.autograd.grad on the loss with create_graph=True.
        adaptive (bool): whether to adaptively select the rank or not
    """
    if not adaptive:
        return _nys_hess_approx(grad_tuple, rank0, _params_list, chunk_size, verbose, adaptive)

    # adaptive rank selection
    l_k         = 999999.
    rank        = rank0
    rank_diff   = int(rank - 0)  # number of new sketch
    ome_T_prev  = None 
    Y_T_prev    = None
    
    while l_k / mu > 10:
        if rank > max_rank:
            rank        = rank - rank_diff  # return to previous
            rank_diff   = max_rank - rank
            rank        = max_rank
        
        U, S, ome_T_prev, Y_T_prev  = _nys_hess_approx(grad_tuple, rank_diff, _params_list, chunk_size, verbose, adaptive, ome_T_prev, Y_T_prev)
        l_k                         = S[-1]
        if verbose:
            print(f'Rank = {rank}, l_k / mu = {l_k / mu}')
        
        if rank == max_rank:
            break
        
        # next_iter
        rank_prev                   = rank
        rank                        = int(rank * rank_mult)         # multiply the rank
        rank_diff                   = rank - rank_prev              # delta rank = rank, so doubles the rank


    return U, S


def _nys_hess_approx(grad_tuple, rank, _params_list, chunk_size, verbose, adaptive = False, ome_T_prev= None, Y_T_prev = None):
    """Update the Nystrom approximation of the Hessian.

    Args:
        grad_tuple (tuple): tuple of Tensors containing the gradients of the loss w.r.t. the parameters. 
            This tuple can be obtained by calling torch.autograd.grad on the loss with create_graph=True.
        adaptive (bool): whether to output the sketch matrix and the test matrix for adaptive rank selection or not
    """

    # Flatten and concatenate the gradients
    gradsH = torch.cat([gradient.reshape(-1)
                        for gradient in grad_tuple if gradient is not None])
    # print(f'gradsH dtype: {gradsH.dtype}')

    # Generate test matrix (NOTE: This is transposed test matrix --> vmap on dim=0)
    p = gradsH.shape[0]
    ome_T = torch.randn(
        (rank, p), device=gradsH.device, dtype=gradsH.dtype) / (p ** 0.5)
    ome_T = torch.linalg.qr(ome_T.t(), mode='reduced')[0].t() # (Q,R)[0] takes Q -- the basis; ome_T.shape: [rank, p] due to t.() twice

    Y_T = _hvp_vmap(gradsH, _params_list, chunk_size)(ome_T) # Y_T.shape: [rank, p]

    if ome_T_prev is not None and Y_T_prev is not None:
        assert ome_T_prev.shape[1] == ome_T.shape[1]
        ome_T   = torch.cat([ome_T_prev, ome_T], dim=0)
        Y_T     = torch.cat([Y_T_prev, Y_T], dim=0)

    with torch.no_grad():
        # Calculate shift
        shift = torch.finfo(Y_T.dtype).eps
        Y_T_shifted = Y_T + shift * ome_T # Y_T_shifted.shape: [rank, p]

        # Calculate ome_T^T * H * ome_T (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_T_shifted, ome_T.t()) # choleskytarget.shape = [rank, rank]

        # Perform Cholesky, if fails, do eigendecomposition
        # The new shift is the abs of smallest eigenvalue (negative) plus the original shift
        try:
            L = torch.linalg.cholesky(choleskytarget) # lower triangular matrix, L.shape = [rank, rank]
            success = True
        except:
            # eigendecomposition, eigenvalues and eigenvector matrix
            eigs, eigvectors = torch.linalg.eigh(choleskytarget)
            # print(f'eigs, eigvectors dtype: {eigs.dtype}, {eigvectors.dtype}')
            
            attempt = 0
            max_attempt = 1
            success = False 
            initial_shift = shift 

            while not success and attempt < max_attempt:
                try:
                    shift = initial_shift + (2 ** attempt) * torch.abs(torch.min(eigs))
                    # add shift to eigenvalues
                    eigs = eigs + shift
                    # print(f'eigs, shift dtype: {eigs.dtype}, {shift.dtype}')
                    
                    # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T
                    L = torch.linalg.cholesky(
                        torch.mm(eigvectors, torch.mm(torch.diag(eigs.to(eigvectors.dtype)), eigvectors.T)))

                    success = True 

                except:
                    attempt += 1
                    # raise RuntimeError("Cholesky decomposition failed after maximum attempts with adjustments.")
        
        if not success:
            print("---Cholesky failed---")
            B = torch.mm(Y_T_shifted.t(), 
                            torch.mm(eigvectors, torch.mm(torch.diag((eigs ** (-0.5)).to(eigvectors.dtype)), eigvectors.T))
                            ).t() # B = L^-1 Y_T =  V * S^(-0.5) * V^T * Y_T_shifted

        else:
            try:
                B = torch.linalg.solve_triangular(
                    L, Y_T_shifted, upper=False, left=True) # B.shape = [rank, p]
            except:
                # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
                B = torch.linalg.solve_triangular(L.to('cpu'), Y_T_shifted.to(
                    'cpu'), upper=False, left=True).to(L.device)


        # B = V * S * U^T b/c we have been using transposed sketch
        _, S, UT = torch.linalg.svd(B, full_matrices=False) # V.shape = [rank, rank], S.shape = [rank], UT.shape = [rank, p]
        # U = UT.t()
        S = torch.max(torch.square(S) - shift, torch.tensor(0.0))
        # rho = S[-1] # smallest eigenvalue of the approximated Hessian


        if verbose:
            print(f'Approximate eigenvalues = {S}')
    
    if not adaptive:
        return UT.t(), S
    else:
        return UT.t(), S, ome_T, Y_T


def _hvp_vmap(grad_params, params, chunk_size):
    return vmap(lambda v: _hvp(grad_params, params, v), in_dims=0, chunk_size=chunk_size)

def _hvp(grad_params, params, v):
    Hv = torch.autograd.grad(grad_params, params, grad_outputs=v.to(grad_params.dtype),
                                retain_graph=True)
    Hv = tuple(Hvi.detach() for Hvi in Hv)
    return torch.cat([Hvi.reshape(-1) for Hvi in Hv])




class NysOpt(Optimizer):
    """Base Class of optimizers that use Nyström approximation.

    This implementation is based on the code of the following paper:
        `Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective.
        Preprint, 2024. <https://arxiv.org/abs/2402.01868>`

        NOTE: This optimizer is currently a beta version. 

        Our implementation is inspired by the PyTorch implementation of `L-BFGS 
        <https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS>`.

    
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    
    The parameters rank and mu will probably need to be tuned for your specific problem.
    If the optimizer is running very slowly, you can try one of the following:
    - Increase the rank (this should increase the accuracy of the Nyström approximation in PCG)
    - Reduce cg_tol (this will allow PCG to terminate with a less accurate solution)
    - Reduce cg_max_iters (this will allow PCG to terminate after fewer iterations)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1.0)
        rank (int, optional): rank of the Nyström approximation (default: 10)
        mu (float, optional): damping parameter (default: 1e-4)
        chunk_size (int, optional): number of Hessian-vector products to be computed in parallel (default: 1)
        line_search_fn (str, optional): either 'armijo' or None (default: None)
        verbose (bool, optional): verbosity (default: False)
    
    """
    def __init__(self, params, defaults = None, lr=1.0, rank=10, mu=1e-4, chunk_size=1,
                 line_search_fn=None, verbose=False):
        if defaults is None:
            defaults = dict(lr=lr, rank=rank, chunk_size=chunk_size, mu=mu, 
                            line_search_fn=line_search_fn, verbose = verbose)
        super(NysOpt, self).__init__(params, defaults)
        self.rank = rank
        self.mu = mu
        self.chunk_size = chunk_size
        self.line_search_fn = line_search_fn
        self.verbose = verbose
        self.U = None
        self.S = None
        self.n_iters = 0

        if len(self.param_groups) > 1:
            raise ValueError(
                "This optimizer doesn't currently support per-parameter options (parameter groups)")

        if self.line_search_fn is not None and self.line_search_fn != 'armijo':
            raise ValueError("This optimizer only supports Armijo line search")

        self._params = self.param_groups[0]['params']
        self._params_list = list(self._params)
        
        self._numel_cache = None


    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns (i) the loss and (ii) gradient w.r.t. the parameters.
            The closure can compute the gradient w.r.t. the parameters by calling torch.autograd.grad on the loss with create_graph=True.
        """
        pass


    def update_preconditioner(self, grad_tuple):
        """Update the Nystrom approximation of the Hessian.

        Args:
            grad_tuple (tuple): tuple of Tensors containing the gradients of the loss w.r.t. the parameters. 
            This tuple can be obtained by calling torch.autograd.grad on the loss with create_graph=True.
        """
        self.U, self.S = _nys_hess_approx(grad_tuple, self.rank, self._params_list, self.chunk_size, self.verbose)


    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].reshape(p.size()), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data



class SketchySGD(NysOpt):
    """Implements SketchySGD. We assume that there is only one parameter group to optimize.

    This implementation is based on the code of the following paper:
        `Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective.
        Preprint, 2024. <https://arxiv.org/abs/2402.01868>`

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rank (int): sketch rank
        mu (float): regularization
        lr (float): learning rate
        weight_decay (float): weight decay parameter
        momentum (float): momentum parameter
        chunk_size (int): number of Hessian-vector products to compute in parallel
        verbose (bool): option to print out eigenvalues of Hessian approximation
    """

    def __init__(self, params, rank=10, mu=0.1, lr=0.001, weight_decay=0.0,
                 momentum=0.0, chunk_size=1, line_search_fn=None, verbose=False, precond_update_freq=1):
        defaults = dict(rank=rank, mu=mu, lr=lr, weight_decay=weight_decay,
                        momentum=momentum, chunk_size=chunk_size, line_search_fn=line_search_fn, verbose=verbose,
                        precond_update_freq=precond_update_freq)
        
        self.momentum = momentum
        self.momentum_buffer = None

        self.precond_update_freq = precond_update_freq
        
        super(SketchySGD, self).__init__(params, defaults, lr=lr, rank=rank, mu=mu, chunk_size=chunk_size,
                 line_search_fn=line_search_fn, verbose=verbose)

        if self.line_search_fn is not None and self.momentum != 0.0:
            raise ValueError(
                "SketchySGD only supports momentum = 0.0 with line search")

        if self.line_search_fn is not None and weight_decay != 0.0:
            raise ValueError(
                "SketchySGD only supports weight_decay = 0.0 with line search")


        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        g = torch.cat([p.grad.view(-1)
                      for group in self.param_groups for p in group['params'] if p.grad is not None])
        g = g.detach()

        # update momentum buffer
        if self.momentum_buffer is None:
            self.momentum_buffer = g
        else:
            self.momentum_buffer = self.momentum * self.momentum_buffer + g

        # one step update
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            weight_decay = group['weight_decay']
            mu = group['mu']

            # calculate the preconditioned search direction
            UTg = torch.mv(self.U.t(), self.momentum_buffer /
                           (1 - self.momentum ** (self.n_iters + 1)))
            dir = torch.mv(self.U, (self.S + mu).reciprocal() * UTg) + (self.momentum_buffer / (
                1 - self.momentum ** (self.n_iters + 1))) / mu - torch.mv(self.U, UTg) / mu

            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure())
                    self._set_param(x)
                    return loss

                # Use -dir for convention
                t = _armijo(obj_func, x_init, g, -dir, group['lr'])
            else:
                t = group['lr']

            self.state[group_idx]['t'] = t

            # update model parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = dir[ls:ls+np].view(p.shape)
                ls += np
                p.data.add_(-(dp + weight_decay * p.data), alpha=t)

        self.n_iters += 1

        return loss



class NysNewtonCG(NysOpt):
    """Implementation of NysNewtonCG, a damped Newton-CG method that uses Nyström preconditioning.
    
    `Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective.
    Preprint, 2024. <https://arxiv.org/abs/2402.01868>`

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    NOTE: This optimizer is currently a beta version. 

    Our implementation is inspired by the PyTorch implementation of `L-BFGS 
    <https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS>`.
    
    The parameters rank and mu will probably need to be tuned for your specific problem.
    If the optimizer is running very slowly, you can try one of the following:
    - Increase the rank (this should increase the accuracy of the Nyström approximation in PCG)
    - Reduce cg_tol (this will allow PCG to terminate with a less accurate solution)
    - Reduce cg_max_iters (this will allow PCG to terminate after fewer iterations)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1.0)
        rank (int, optional): rank of the Nyström approximation (default: 10)
        mu (float, optional): damping parameter (default: 1e-4)
        chunk_size (int, optional): number of Hessian-vector products to be computed in parallel (default: 1)
        cg_tol (float, optional): tolerance for PCG (default: 1e-16)
        cg_max_iters (int, optional): maximum number of PCG iterations (default: 1000)
        line_search_fn (str, optional): either 'armijo' or None (default: None)
        verbose (bool, optional): verbosity (default: False)
    
    """

    def __init__(self, params, lr=1.0, rank=10, mu=1e-4, chunk_size=1,
                 cg_tol=1e-16, cg_max_iters=1000, line_search_fn=None, verbose=False, 
                 precond_update_freq = 1):
        defaults = dict(lr=lr, rank=rank, chunk_size=chunk_size, mu=mu, cg_tol=cg_tol,
                        cg_max_iters=cg_max_iters, line_search_fn=line_search_fn, precond_update_freq= precond_update_freq)
        
        super(NysNewtonCG, self).__init__(params, defaults, lr=lr, rank=rank, mu=mu, chunk_size=chunk_size,
                 line_search_fn=line_search_fn, verbose=verbose)

        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.precond_update_freq = precond_update_freq
        

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns (i) the loss and (ii) gradient w.r.t. the parameters.
            The closure can compute the gradient w.r.t. the parameters by calling torch.autograd.grad on the loss with create_graph=True.
        """

        # NOTE: The closure must return both the loss and the gradient
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        g = torch.cat([grad.reshape(-1) for grad in grad_tuple if grad is not None])
        
        if self.n_iters == 0:
            # Store the previous direction for warm starting PCG
            self.old_dir = torch.zeros(
                self._numel(), device=self._params[0].device, dtype = g.dtype)

        # One step update
        for group_idx, group in enumerate(self.param_groups):
            def hvp_temp(x):
                return _hvp(g, self._params_list, x)

            # Calculate the Newton direction
            d = _nystrom_pcg(hvp_temp, g, self.old_dir,
                             self.mu, self.U, self.S, self.rank, self.cg_tol, self.cg_max_iters)

            # Store the previous direction for warm starting PCG
            self.old_dir = d

            # Check if d is a descent direction
            if torch.vdot(d, g).real <= 0:
                print("Warning: d is not a descent direction")

            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure()[0])
                    self._set_param(x)
                    return loss

                # Use -d for convention
                t = _armijo(obj_func, x_init, g, -d, group['lr'])
            else:
                t = group['lr']

            self.state[group_idx]['t'] = t

            # update parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = d[ls:ls+np].reshape(p.shape)
                ls += np
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss, g
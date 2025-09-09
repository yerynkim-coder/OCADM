import casadi as ca
import numpy as np
import csv
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

class GenerateData:

    def __init__(self, p_range=[(-2, 2)], p_weights=None, num_samples=100, case=None, param: float = None):
        self.p_range = p_range if isinstance(p_range[0], tuple) else [p_range]
        self.num_samples = num_samples
        self.p_weights = p_weights  # None by default
        self.case = case if case is not None else np.random.randint(1, 5)
        self.param = param

        self.noise_mean = 0.0
        self.noise_std = 0.0

        self.p_current = None
        self.h_current = None

        self._build_symbolic_function()


    def _sample_p(self):
        n_intervals = len(self.p_range)

        if self.p_weights is not None:
            assert len(self.p_weights) == n_intervals, "Length of p_weights must match number of intervals"
            norm_weights = np.array(self.p_weights) / np.sum(self.p_weights)
        else:
            # Default: use interval length as implicit weight
            lengths = [end - start for (start, end) in self.p_range]
            norm_weights = np.array(lengths) / sum(lengths)

        allocated = [int(self.num_samples * w) for w in norm_weights]
        allocated[-1] += self.num_samples - sum(allocated)  # Ensure total sums to num_samples

        p_list = []
        for (start, end), n in zip(self.p_range, allocated):
            p_list.append(np.random.uniform(start, end, n))

        return np.concatenate(p_list).reshape(-1, 1)

    def _symbolic_h(self, p, case):
        if case == 1:
            h = 0
        elif case == 2:
            param = self.param if self.param is not None else 18
            h = (ca.pi * p) / param
        elif case == 3:
            param = self.param if self.param is not None else 0.005
            h = param * ca.cos(18 * p)
        elif case == 4:
            condition_left = p <= -ca.pi / 2
            condition_right = p >= ca.pi / 6
            h_center = ca.sin(3 * p)
            h_flat = 1
            h = ca.if_else(condition_left, h_flat, ca.if_else(condition_right, h_flat, h_center))
        else:
            raise ValueError(f"Invalid slope case: {case}")
        return h

    def _build_symbolic_function(self):
        p_sym = ca.MX.sym("p")
        h_sym = self._symbolic_h(p_sym, self.case)
        self.h_func = ca.Function("h_func", [p_sym], [h_sym])

    def get_symbolic_function(self):
        if self.case is None:
            raise ValueError("Case is not set. Please set a case before getting the symbolic function.")
        return self.h_func

    def update_case(self, new_case):
        self.case = new_case
        self._build_symbolic_function()

    def set_noise(self, mean=0.0, std=0.0):
        self.noise_mean = mean
        self.noise_std = std

    def generate_data(self):
        self.p_current = self._sample_p()
        self.h_current = np.array([float(self.h_func(p)) for p in self.p_current]).reshape(-1, 1)

        if self.noise_std > 0:
            noise = np.random.normal(self.noise_mean, self.noise_std, size=self.num_samples).reshape(-1, 1)
            self.h_current += noise

        return self.p_current, self.h_current

    def save_to_csv(self, filename):
        if self.p_current is None or self.h_current is None:
            raise ValueError("No data available. Please run generate_data() first.")
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['p', 'h'])
            for p_val, h_val in zip(self.p_current, self.h_current):
                writer.writerow([float(p_val), float(h_val)])

    def load_from_csv(self, filename):
        p_list, h_list = [], []
        with open(filename, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                p_list.append(float(row[0]))
                h_list.append(float(row[1]))
        self.p_current = np.array(p_list).reshape(-1, 1)
        self.h_current = np.array(h_list).reshape(-1, 1)
        return self.p_current, self.h_current
    
    def plot(self, show_true_func=True, title="Generated Dataset and True Function", ax=None):
        if self.p_current is None or self.h_current is None:
            raise ValueError("No data to plot. Please call generate_data() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        # Plot noisy training data
        ax.plot(self.p_current, self.h_current, 'k.', label='Training Data', alpha=0.5)

        # Plot true function over all defined intervals
        if show_true_func:
            # Compute overall range from all intervals
            p_min_global = min(start for (start, _) in self.p_range)
            p_max_global = max(end for (_, end) in self.p_range)

            # Generate dense p values across the full range
            p_dense = np.linspace(p_min_global, p_max_global, 1000).reshape(-1, 1)
            h_dense = np.array([float(self.h_func(p)) for p in p_dense]).reshape(-1, 1)
            ax.plot(p_dense, h_dense, 'b--', linewidth=2)

            ax.plot([], [], 'b--', label="True Function")  # for legend only

        ax.set_xlabel("p")
        ax.set_ylabel("h(p)")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    


class Identifier_LR:
    def __init__(self, basis_functions):
        """
        basis_functions: list of callables, each taking a (N,1) array and returning a (N,1) array
        """
        self.basis_functions = basis_functions
        self.theta = None
        self.p_train = None
        self.h_train = None
    
    def reset(self):
        self.theta = None
        self.p_train = None
        self.h_train = None

    def fit(self, p, h):
        p = p.reshape(-1, 1)
        h = h.reshape(-1, 1)
        Phi = np.hstack([f(p) for f in self.basis_functions])  # N x D
        self.theta = np.linalg.lstsq(Phi, h, rcond=None)[0]    # D x 1
        self.p_train = p
        self.h_train = h

    def get_params(self):
        if self.theta is None:
            raise ValueError("Model has not been fitted yet.")
        return self.theta

    def predict(self, p):
        p = p.reshape(-1, 1)
        Phi = np.hstack([f(p) for f in self.basis_functions])  # N x D
        return Phi @ self.theta

    def plot(self, p_test=None, true_func=None, title='Linear Regression', ax=None):
        """
        p_test: (N, 1) test points for drawing smooth fitted curve
        true_func: callable, true slope function h(p)
        ax: optional matplotlib.axes.Axes to plot on
        """
        if self.p_train is None or self.h_train is None:
            raise ValueError("You must fit the model before plotting.")

        if p_test is None:
            p_test = np.linspace(self.p_train.min(), self.p_train.max(), 300).reshape(-1, 1)

        h_pred = self.predict(p_test)

        # Use passed-in axis or create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(title)

        # True function
        if true_func is not None:
            h_true = np.array([float(true_func(p)) for p in p_test]).reshape(-1, 1)
            ax.plot(p_test, h_true, 'b--', linewidth=2, label='True Function')

        # Training data
        if len(self.p_train)<50:
            ax.plot(self.p_train, self.h_train, 'k.', label='Training Data', alpha=0.5, markersize=15)
        elif len(self.p_train)<200:
            ax.plot(self.p_train, self.h_train, 'k.', label='Training Data', alpha=0.5, markersize=10)
        else:
            ax.plot(self.p_train, self.h_train, 'k.', label='Training Data', alpha=0.5)

        # Prediction
        ax.plot(p_test, h_pred, color='red', linewidth=2, label='Fitted Curve (LR)')

        ax.set_xlabel('p')
        ax.set_ylabel('h(p)')
        ax.grid(True)
        ax.legend()
    


class Identifier_BLR(Identifier_LR):
    def __init__(self, basis_functions, sigma2=1.0, mu0=None, Sigma0=None):
        """
        Parameters:
        - basis_functions: list of basis functions
        - sigma2: observation noise variance σ²
        - mu0: prior mean vector μ₀ (default: 0 vector)
        - Sigma0: prior covariance matrix Σ₀ (default: λ² I)
        """
        super().__init__(basis_functions)
        self.sigma2 = sigma2
        self.mu0_user = mu0
        self.Sigma0_user = Sigma0
        self.mu_theta = None
        self.Sigma_theta = None

    def reset(self):
        super().reset()
        self.mu_theta = None
        self.Sigma_theta = None

    def fit(self, p, h):
        p = p.reshape(-1, 1)
        h = h.reshape(-1, 1)
        Phi = np.hstack([f(p) for f in self.basis_functions])  # N x B
        B = Phi.shape[1]

        # Set prior mean and covariance
        mu0 = self.mu0_user if self.mu0_user is not None else np.zeros((B, 1))
        Sigma0 = self.Sigma0_user if self.Sigma0_user is not None else np.eye(B)

        # Compute posterior
        Sigma0_inv = np.linalg.inv(Sigma0)
        self.Sigma_theta = np.linalg.inv(Sigma0_inv + (1 / self.sigma2) * Phi.T @ Phi)
        self.mu_theta = self.Sigma_theta @ (Sigma0_inv @ mu0 + (1 / self.sigma2) * Phi.T @ h)

        self.theta = self.mu_theta  # for consistency with LR interface
        self.p_train = p
        self.h_train = h

    def predict(self, p):
        p = p.reshape(-1, 1)
        Phi_test = np.hstack([f(p) for f in self.basis_functions])  # N x B

        mean = Phi_test @ self.mu_theta
        var = np.sum(Phi_test @ self.Sigma_theta * Phi_test, axis=1, keepdims=True) + self.sigma2
        std = np.sqrt(var)
        return mean, std

    def plot(self, p_test=None, true_func=None, title='Bayesian Linear Regression', ax=None, larger_dot=False):
        if self.p_train is None or self.h_train is None:
            raise ValueError("You must fit the model before plotting.")

        if p_test is None:
            p_test = np.linspace(self.p_train.min(), self.p_train.max(), 300).reshape(-1, 1)

        y_mean, y_std = self.predict(p_test)

        # use provided ax or create one
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(title)

        # True function
        if true_func is not None:
            y_true = np.array([float(true_func(p)) for p in p_test]).reshape(-1, 1)
            ax.plot(p_test, y_true, 'b--', label='True function', linewidth=2)

        # Training data
        if len(self.p_train)<50:
            ax.plot(self.p_train, self.h_train, 'k.', label='Training Data', alpha=0.5, markersize=15)
        elif len(self.p_train)<200:
            ax.plot(self.p_train, self.h_train, 'k.', label='Training Data', alpha=0.5, markersize=10)
        else:
            ax.plot(self.p_train, self.h_train, 'k.', label='Training Data', alpha=0.5)

        # Prediction mean
        ax.plot(p_test, y_mean, color='red', label='Model prediction (BLR)', linewidth=2)

        # Confidence intervals
        for i in range(1, 4):
            ax.fill_between(
                p_test.ravel(),
                (y_mean - i * y_std).ravel(),
                (y_mean + i * y_std).ravel(),
                color='coral',
                alpha=0.15 if i == 3 else 0.3,
                label='Model uncertainty (3 std.)' if i == 3 else None
            )

        ax.set_xlabel('p')
        ax.set_ylabel('h(p)')
        ax.grid(True)
        ax.legend()
        

# Move this function outside the class
def plot_param_over_sample_size(p, 
                                h, 
                                lr_model=None, 
                                blr_model=None, 
                                sample_indices=None, 
                                plot_dims=None,
                                groundtruth=None, 
                                title=None):
    """
    Compare LR and BLR parameter convergence on subsets of the same (p, h) dataset.

    Parameters:
    - lr_model: instance of Identifier_LR
    - blr_model: instance of Identifier_BLR
    - p, h: full training data
    - sample_indices: list or array of specific sample sizes to evaluate and plot (e.g., [10, 100, 1000]).
                        If None, use default log-scale sample sizes.
    """

    N_total = p.shape[0]

    if lr_model:
        D = len(lr_model.basis_functions)
    elif blr_model:
        D = len(blr_model.basis_functions)
    else:
        raise ValueError("At least one model (LR or BLR) must be provided.")

    # Determine sample sizes
    if sample_indices is None:
        sample_sizes = np.unique(np.logspace(np.log10(5), np.log10(N_total), num=9, dtype=int))
    else:
        sample_sizes = np.array([i for i in sample_indices if i <= N_total])
        if len(sample_sizes) == 0:
            raise ValueError("All specified sample indices exceed the dataset size.")
    
    # Determine which dimensions to plot
    if plot_dims is None:
        plot_dims = list(range(D))
    else:
        plot_dims = [i for i in plot_dims if 0 <= i < D]
        if not plot_dims:
            raise ValueError("No valid plot_dims remain after filtering.")

    lr_thetas = np.zeros((len(sample_sizes), D))
    blr_means = np.zeros((len(sample_sizes), D))
    blr_stds = np.zeros((len(sample_sizes), D))

    for i, N in enumerate(sample_sizes):
        p_sub = p[:N]
        h_sub = h[:N]

        # Linear regression
        if lr_model:
            lr_model.reset()
            lr_model.fit(p_sub, h_sub)
            lr_thetas[i] = lr_model.get_params().flatten()

        # Bayesian linear regression
        if blr_model:
            blr_model.reset()
            blr_model.fit(p_sub, h_sub)
            blr_means[i] = blr_model.mu_theta.flatten()
            blr_stds[i] = np.sqrt(np.diag(blr_model.Sigma_theta))

    # Plot
    fig, axes = plt.subplots(len(plot_dims), 1, figsize=(8, 3 * len(plot_dims)))
    if len(plot_dims) == 1:
        axes = [axes]

    for idx, dim in enumerate(plot_dims):
        ax = axes[idx]
        if lr_model:
            ax.plot(sample_sizes, lr_thetas[:, dim], 'o-', color='green', label='LR estimate', alpha=0.7)
        if blr_model:
            ax.plot(sample_sizes, blr_means[:, dim], 'r-o', label='BLR mean', linewidth=2)
            ax.fill_between(sample_sizes,
                            blr_means[:, dim] - 3 * blr_stds[:, dim],
                            blr_means[:, dim] + 3 * blr_stds[:, dim],
                            color='red', alpha=0.2,
                            label='BLR ±3 Std')
        if groundtruth is not None:
            ax.axhline(y=groundtruth[dim], color='b', linestyle='--', label='Ground Truth')
        ax.set_ylabel(f'$\\theta_{{{dim}}}$')
        ax.set_xscale('log')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Dataset Size (log scale)")
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("LR vs BLR Parameter Convergence")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_rmse_over_sample_size(p, h, model, true_func, test_range=np.array([-2.0, 2.0]), sample_indices=None, title=None):
    """
    Plot the RMSE of a fitted model against the true function as a function of dataset size.

    Parameters:
    - p, h: full training data
    - model: any model with fit(p, h) and predict(p) methods
    - true_func: callable function that returns true h given p
    - sample_indices: optional list of sample sizes to evaluate
    - title: optional plot title
    """

    N_total = p.shape[0]

    p_test = np.linspace(test_range[0], test_range[1], 300).reshape(-1, 1)

    # Determine sample sizes
    if sample_indices is None:
        sample_sizes = np.unique(np.logspace(np.log10(5), np.log10(N_total), num=9, dtype=int))
    else:
        sample_sizes = np.array([i for i in sample_indices if i <= N_total])
        if len(sample_sizes) == 0:
            raise ValueError("All specified sample indices exceed the dataset size.")

    rmses = []

    for N in sample_sizes:
        p_sub = p[:N]
        h_sub = h[:N]

        model.reset()
        model.fit(p_sub, h_sub)
        h_pred = model.predict(p_test)

        h_true = np.array([float(true_func(p_i)) for p_i in p_test]).reshape(-1, 1)
        rmse = np.sqrt(np.mean((h_pred.flatten() - h_true.flatten()) ** 2))
        rmses.append(rmse)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, rmses, marker='o', linewidth=2)
    plt.xscale('log')
    plt.xlabel("Dataset Size (log scale)")
    plt.ylabel("RMSE against Ground Truth")
    plt.grid(True)
    plt.title(title or "Model RMSE vs. Dataset Size")
    plt.tight_layout()
    plt.show()
    


def animate_training_progress(p, h, lr_model=None, blr_model=None, sample_indices=None, true_func=None, title='Model Fitting Animation'):
    """
    Animate model fitting results for LR and/or BLR across increasing sample sizes.
    Ensures consistent axes across all frames and subplots.
    """
    if not lr_model and not blr_model:
        raise ValueError("At least one of lr_model or blr_model must be provided.")

    N_total = p.shape[0]
    if sample_indices is None:
        sample_indices = np.unique(np.logspace(np.log10(2), np.log10(N_total), num=9, dtype=int))
    else:
        sample_indices = [n for n in sample_indices if n <= N_total]
        if len(sample_indices) == 0:
            raise ValueError("All sample_indices exceed dataset size.")

    has_lr = lr_model is not None
    has_blr = blr_model is not None

    fig, axes = plt.subplots(1, (has_lr + has_blr), figsize=(10, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]  # force list

    # --- Animation frame update ---
    def update(frame_idx):

        n = sample_indices[frame_idx]
        p_sub, h_sub = p[:n], h[:n]

        fig.suptitle(f"{title}\nSample size = {n}", fontsize=14)

        p_test_uniform = np.linspace(p.min(), p.max(), 300).reshape(-1, 1)
        y_values = [h_sub]

        if has_lr:
            lr_model.reset()
            lr_model.fit(p_sub, h_sub)
            y_pred_lr = lr_model.predict(p_test_uniform)
            y_values.append(y_pred_lr)

        if has_blr:
            blr_model.reset()
            blr_model.fit(p_sub, h_sub)
            mu, std = blr_model.predict(p_test_uniform)
            y_values.append(mu + 3 * std)
            y_values.append(mu - 3 * std)

        # Compute per-frame ylim
        y_concat = np.vstack(y_values)
        y_margin = 0.05 * (y_concat.max() - y_concat.min())
        ylim = (y_concat.min() - y_margin, y_concat.max() + y_margin)

        # Plot LR
        if has_lr:
            ax_lr = axes[0] if has_blr else axes[0]
            ax_lr.clear()
            ax_lr.set_xlim(p.min(), p.max())
            ax_lr.set_ylim(ylim)
            # True function
            if true_func is not None:
                y_true = np.array([float(true_func(p)) for p in p_test_uniform]).reshape(-1, 1)
                ax_lr.plot(p_test_uniform, y_true, 'b--', label='True function', linewidth=2)
            lr_model.plot(p_test=p_test_uniform, ax=ax_lr)

        # Plot BLR
        if has_blr:
            ax_blr = axes[1] if has_lr else axes[0]
            ax_blr.clear()
            ax_blr.set_xlim(p.min(), p.max())
            ax_blr.set_ylim(ylim)
            # True function
            if true_func is not None:
                y_true = np.array([float(true_func(p)) for p in p_test_uniform]).reshape(-1, 1)
                ax_blr.plot(p_test_uniform, y_true, 'b--', label='True function', linewidth=2)
            blr_model.plot(p_test=p_test_uniform, ax=ax_blr)

    anim = FuncAnimation(fig, update, frames=len(sample_indices), interval=1500, repeat=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())

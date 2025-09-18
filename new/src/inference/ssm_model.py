import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
import numpy as np
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from pathlib import Path

import logging
logging.getLogger("pymc").setLevel(logging.ERROR)

def logit(p):
    return np.log(p) - np.log(1 - p)

def invlogit(x):
    return 1 / (1 + np.exp(-x))

def plot_p_timeseries(p_t_inferred, p_t_true=None, timesteps=None, 
                     is_dummy_data=True, save_filename=None):
    """
    Plot time series of inferred p values and optionally compare with true values.
    
    Args:
        p_t_inferred (array): Inferred p_t series
        p_t_true (array, optional): True p_t series for comparison
        timesteps (array, optional): Time step indices
        is_dummy_data (bool): If True, save to figures/exploratory/, else figures/publication/
        save_filename (str, optional): Custom filename. If None, auto-generates based on data type.
    
    Returns:
        str: Path to saved plot file
    """
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Generate timesteps if not provided
    if timesteps is None:
        timesteps = np.arange(len(p_t_inferred))
    
    # Plot inferred p_t series
    plt.plot(timesteps, p_t_inferred, 'b-', linewidth=2, marker='o', 
             markersize=6, label='Inferred p_t', alpha=0.8)
    
    # Plot true p_t series if provided
    if p_t_true is not None:
        plt.plot(timesteps, p_t_true, 'r--', linewidth=2, marker='s', 
                 markersize=6, label='True p_t', alpha=0.8)
        
        # Calculate and display correlation
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(p_t_inferred, p_t_true)
        mae = np.mean(np.abs(p_t_inferred - p_t_true))
        
        plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}\nMAE: {mae:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Formatting
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('p_t (Probability)', fontsize=12)
    plt.title('Time Series of Environmental Probability (p_t)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Set up save directory and filename
    if is_dummy_data:
        save_dir = Path("figures/exploratory")
    else:
        save_dir = Path("figures/publication")
    
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if save_filename is None:
        if is_dummy_data:
            save_filename = "p_timeseries_dummy_data.pdf"
        else:
            save_filename = "p_timeseries_real_data.pdf"
    
    # Ensure filename ends with .pdf
    if not save_filename.endswith('.pdf'):
        save_filename += '.pdf'
    
    # Full path
    save_path = save_dir / save_filename
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ p_t time series plot saved to: {save_path}")
    return str(save_path)

def fit_ssm_random_walk(y_values, l_t_series, init_x_traj, n_samples=1000):
    """
    Fits a State-Space Model where the latent belief x follows a Gaussian Random Walk.

    Args:
        y_values (np.ndarray): Time series of observed log-income growth.
        l_t_series (np.ndarray): Time series of the number of outcomes, l.
        init_x_traj (np.ndarray): An initial guess for the x trajectory, used for initialization.
        n_samples (int): Number of samples for the VI fit.

    Returns:
        tuple: (arviz.InferenceData, float) containing the fitted model and loss.
    """
    with pm.Model() as model:
        # --- Priors ---
        # Volatility of the belief evolution. A larger sigma allows faster belief changes.
        # Increased from 0.1 to 0.3 to better capture agents with strong learning patterns.
        sigma = pm.HalfNormal("sigma", 0.3)
        
        # --- Latent State: Belief (x) ---
        # The agent's belief is modeled as a random walk in logit-space to keep it in (0,1).
        # We initialize it with a Normal distribution centered on our smart guess.
        z_init_dist = pm.Normal.dist(mu=logit(init_x_traj[0]), sigma=0.1)
        z = pm.GaussianRandomWalk("z", sigma=sigma, init_dist=z_init_dist, shape=len(y_values))
        
        # Transform the latent state z back to the probability space [0,1] for x
        x = pm.Deterministic("x", pm.math.invlogit(z))

        # --- Observation Model ---
        # The expected log-income growth (y) depends on the belief x.
        # We need to handle the two cases: wins (y > 0) and losses (y < 0).
        mu_win = pt.log(l_t_series * x)
        mu_loss = pt.log(l_t_series * (1 - x))

        # We use pt.switch to select the correct formula based on the sign of the observed y.
        # This is a robust way to handle the discontinuity.
        # Note: We can't use y_values directly in the switch condition inside the model.
        # Instead, we pass the sign of y_values as a constant.
        y_signs = np.sign(y_values)
        mu = pt.switch(pt.gt(y_signs, 0), mu_win, mu_loss)
        
        # Observation noise - how much the actual income deviates from the theoretical expectation.
        obs_sigma = pm.HalfNormal("obs_sigma", 0.5)

        # Likelihood of observing the income data given our model
        y_obs = pm.Normal("y_obs", mu=mu, sigma=obs_sigma, observed=y_values)

        # --- Inference ---
        approx = pm.fit(n=n_samples, method='advi', obj_optimizer=pm.adagrad(learning_rate=0.1))
        idata = approx.sample(1000)
        loss = -approx.hist[-1]

    return idata, loss

def fit_ssm_dynamic_p_model(y, p_t_series, l=2, method='vi', vi_steps=15000,
                            z0_mu=0.0, z0_sigma=1.0, k_prior_sigma=0.5, 
                            tau_sigma=0.1, obs_sigma_prior=0.05, 
                            draws=1000, tune=1000, chains=4, 
                            target_accept=0.98, random_seed=42):
    
    y = np.asarray(y, dtype=float)
    p_t_series = np.asarray(p_t_series, dtype=float)
    T = len(y)

    coords = {"time": np.arange(T)}
    
    with pm.Model(coords=coords) as model:
        # --- Priors ---
        # Agent's learning rate
        k = pm.HalfNormal("k", sigma=k_prior_sigma)
        # Agent's initial belief state (in logit-space)
        z0 = pm.Normal("z0", mu=z0_mu, sigma=z0_sigma)
        # Standard deviation of the process noise
        tau = pm.HalfNormal("tau", sigma=tau_sigma)
        # Standard deviation for the likelihood
        sigma = pm.HalfNormal("sigma", sigma=obs_sigma_prior)
        
        # Convert p_t to logit space to be the target for z_t
        p_t_logit = pt.log(p_t_series / (1 - p_t_series))
        
        # --- Recurrent Evolution of Belief State (z_t) ---
        def step(t, z_prev, p_target_logit, k_rate, noise):
            # Your proposed evolution equation
            z_mean = (p_target_logit * (1 / (k_rate * l)) + z_prev) / (1 + 1 / (k_rate * l))
            # The new state is the mean + noise
            z_t = z_mean + noise[t]
            return z_t

        # This is noise for the *entire* time series, pre-supplied to scan
        process_noise = pm.Normal("process_noise", mu=0, sigma=tau, dims="time")

        z_full, _ = scan(
            fn=step,
            sequences=[pt.arange(T), p_t_logit],
            outputs_info=[z0],
            non_sequences=[k, process_noise],
            strict=True
        )
        
        # --- Likelihood ---
        x = pm.Deterministic("x", pm.math.invlogit(z_full), dims="time")
        
        mu_correct = pm.Deterministic("mu_correct", pm.math.log(l * x), dims="time")
        mu_wrong = pm.Deterministic("mu_wrong", pm.math.log(l * (1 - x)), dims="time")
        
        comp_correct = pm.Normal.dist(mu=mu_correct, sigma=sigma)
        comp_wrong = pm.Normal.dist(mu=mu_wrong, sigma=sigma)
        
        # The weight `w` is now the dynamic p_t_series data
        pm.Mixture("y_obs", w=pt.stack([p_t_series, 1 - p_t_series]).T, 
                   comp_dists=[comp_correct, comp_wrong], observed=y, dims="time")
        
        # --- Sampling ---
        if method == 'vi':
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                approx = pm.fit(n=vi_steps, random_seed=random_seed, progressbar=False)
            idata = approx.sample(draws=draws, random_seed=random_seed)
            loss = approx.hist[-1]
        else: # MCMC
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, 
                              random_seed=random_seed, progressbar=False)
            loss = None
            
        return idata, loss 

def fit_l_cross_sectional(y_values_t, p_hat_t, population_weights, delta=0.05, n_samples=10000):
    """
    Cross-sectional Bayesian model for estimating l at a single timestep.

    We evaluate a grid of l values and compute a pseudo-likelihood that the implied
    agent beliefs x come from a Beta distribution. Importantly, each observed y can
    arise from either a win or a loss branch:
      y = log(l * x)          (win)
      y = log(l * (1 - x))    (loss)
    So for a given y and l, there are two candidate x's:
      x_win  = exp(y) / l
      x_loss = 1 - exp(y) / l
    We marginalize over this latent win/loss with weights p_hat_t and (1 - p_hat_t).

    NOTE: We intentionally do NOT include a Jacobian factor here because we are
    comparing l values via a pseudo-likelihood in x-space; including |dx/dy| adds
    a 1/l factor that systematically biases the estimate downward in this setting.
    """
    # Convert to numpy array and filter out invalid values
    y_values_t = np.array(y_values_t)
    population_weights = np.array(population_weights)

    valid_mask = np.isfinite(y_values_t)
    y_values_clean = y_values_t[valid_mask]
    weights_clean = population_weights[valid_mask]

    if len(y_values_clean) < 5:
        raise ValueError(f"Need at least 5 valid data points, got {len(y_values_clean)}")

    # Normalize weights
    if np.sum(weights_clean) > 0:
        # Normalize weights to sum to number of observations (n) to maintain proper likelihood scaling
        # This preserves relative importance while avoiding numerical issues
        n_obs = len(weights_clean)
        normalized_weights = (weights_clean / np.sum(weights_clean)) * n_obs
    else:
        normalized_weights = np.ones_like(weights_clean)


    # Grid of l values
    l_grid = np.linspace(1, 4, 2000)
    log_likelihoods = []

    # Prior on beliefs x ~ Beta(alpha, beta)
    # Keep a modestly informative prior centered at 0.6 with low concentration
    mu = 0.57
    kappa_fixed = 3.0
    alpha = mu * kappa_fixed
    beta = (1 - mu) * kappa_fixed

    from scipy.stats import beta as beta_dist
    from scipy.special import logsumexp

    eps = 1e-6

    for l_test in l_grid:
        # Two candidate x's for each y
        x_win = np.clip(np.exp(y_values_clean) / l_test, eps, 1 - eps)
        x_loss = np.clip(1 - (l_test - 1) * np.exp(y_values_clean) / l_test, eps, 1 - eps)

        # Log-prob under Beta prior for both branches
        logpdf_win = beta_dist.logpdf(x_win, alpha, beta)
        logpdf_loss = beta_dist.logpdf(x_loss, alpha, beta)

        # Mixture over branches with weights p_hat_t and (1 - p_hat_t)
        # log( p * exp(logpdf_win) + (1-p) * exp(logpdf_loss) ) via logsumexp
        comp = np.vstack((np.log(max(p_hat_t, eps)) + logpdf_win,
                          np.log(max(1 - p_hat_t, eps)) + logpdf_loss))
        log_like_per_obs = logsumexp(comp, axis=0)
        
        # Apply population weights to the log-likelihood
        log_likelihood = np.sum(log_like_per_obs * normalized_weights)
        log_likelihoods.append(log_likelihood)

    log_likelihoods = np.array(log_likelihoods)
    if np.all(np.isinf(log_likelihoods)):
        raise ValueError("All likelihood values are invalid")

    # Normalize to probabilities
    max_log_lik = np.max(log_likelihoods[np.isfinite(log_likelihoods)])
    normalized_log_lik = log_likelihoods - max_log_lik
    probabilities = np.exp(normalized_log_lik)
    probabilities = probabilities / np.sum(probabilities)

    # Discrete posterior samples of l
    n_posterior_samples = 1000
    posterior_indices = np.random.choice(len(l_grid), size=n_posterior_samples, p=probabilities)
    l_posterior_samples = l_grid[posterior_indices]

    class MockLVariable:
        def __init__(self, samples):
            self.values = samples.reshape(1, 1, -1)
        def flatten(self):
            return self.values.flatten()

    class MockPosterior:
        def __init__(self, l_samples):
            self._l_var = MockLVariable(l_samples)
        def __getitem__(self, key):
            if key == "l":
                return self._l_var
            else:
                raise KeyError(f"Key '{key}' not found")

    class MockInferenceData:
        def __init__(self, l_samples):
            self.posterior = MockPosterior(l_samples)

    return MockInferenceData(l_posterior_samples)


def estimate_l_time_series_bayesian(dummy_data, delta=0.05, n_samples=8000, 
                                   rolling_window=3):
    """
    Estimates l for each timestep using the cross-sectional Bayesian approach,
    then applies temporal smoothing.
    
    This implements the full "Best of Both Worlds" workflow:
    1. For each timestep, run cross-sectional VI to get l_est_t
    2. Apply rolling mean smoothing to the time series of l estimates
    
    Args:
        dummy_data: The dummy data dictionary containing agent growth rates
        delta (float): Sub-optimality offset for agent beliefs
        n_samples (int): Number of VI samples per timestep
        rolling_window (int): Window size for temporal smoothing
        
    Returns:
        dict: Dictionary mapping timestep -> smoothed l estimate
    """
    
    # Extract data
    vi_data = dummy_data['vi_data']
    p_t_series = dummy_data['p_t_series']
    
    print(f"Running Bayesian l estimation for {len(p_t_series)} timesteps...")
    
    # Step 1: Collect all growth rates by timestep
    growth_rates_by_timestep = {}
    population_by_timestep = {} # Store populations as well

    for agent_data in vi_data:
        y_values = agent_data['income_growth_rates']
        populations = agent_data.get('population', np.ones_like(y_values)) # Default to 1 if not present

        for i, (y, pop) in enumerate(zip(y_values, populations)):
            if np.isfinite(y):
                timestep = i
                if timestep not in growth_rates_by_timestep:
                    growth_rates_by_timestep[timestep] = []
                    population_by_timestep[timestep] = []
                growth_rates_by_timestep[timestep].append(y)
                population_by_timestep[timestep].append(pop)
    
    # Step 2: Run cross-sectional VI for each timestep
    l_estimates = {}
    
    for timestep in range(len(p_t_series)):
        if timestep in growth_rates_by_timestep:
            y_values_t = growth_rates_by_timestep[timestep]
            p_hat_t = p_t_series[timestep]
            pop_weights_t = population_by_timestep[timestep]
            
            try:
                # Run the cross-sectional Bayesian model with weights
                idata = fit_l_cross_sectional(y_values_t, p_hat_t, pop_weights_t, delta, n_samples)
                
                # Extract the posterior mean as our point estimate
                l_posterior_samples = idata.posterior["l"].values.flatten()
                l_est_t = np.mean(l_posterior_samples)
                l_estimates[timestep] = l_est_t
                
                print(f"  Timestep {timestep}: l_est = {l_est_t:.3f} "
                      f"(from {len(y_values_t)} agents)")
                
            except Exception as e:
                print(f"  Timestep {timestep}: Failed to estimate l ({str(e)})")
                # Use a reasonable default if estimation fails
                l_estimates[timestep] = 2.5
    
    # Step 3: Apply temporal smoothing (rolling mean)
    if len(l_estimates) >= rolling_window:
        print(f"\nApplying temporal smoothing (rolling window = {rolling_window})...")
        
        # Convert to sorted lists for rolling mean calculation
        sorted_timesteps = sorted(l_estimates.keys())
        l_values = [l_estimates[t] for t in sorted_timesteps]
        
        # Apply rolling mean
        l_smoothed_values = []
        for i in range(len(l_values)):
            # Define the window around the current point
            start_idx = max(0, i - rolling_window // 2)
            end_idx = min(len(l_values), i + rolling_window // 2 + 1)
            window_values = l_values[start_idx:end_idx]
            l_smoothed = np.mean(window_values)
            l_smoothed_values.append(l_smoothed)
        
        # Create the final smoothed dictionary
        l_smoothed_dict = {}
        for i, timestep in enumerate(sorted_timesteps):
            l_smoothed_dict[timestep] = l_smoothed_values[i]
            print(f"  Timestep {timestep}: l_raw = {l_estimates[timestep]:.3f} -> "
                  f"l_smooth = {l_smoothed_dict[timestep]:.3f}")
        
        return l_smoothed_dict
    
    else:
        print(f"Not enough timesteps for smoothing ({len(l_estimates)} < {rolling_window})")
        return l_estimates

def fit_l_and_x_hierarchical(y_values_t, p_hat_t, population_weights=None, n_samples=2000):
    """
    Infers l and the distribution of agent beliefs x simultaneously using a hierarchical Bayesian model.

    This model treats 'l' as a global parameter and agent beliefs 'x' as latent variables
    drawn from a common Beta distribution, whose parameters are also inferred.

    Args:
        y_values_t (np.ndarray): Array of observed log-income growth for a single timestep.
        p_hat_t (float): The estimated probability of a "win" at this timestep.
        population_weights (np.ndarray, optional): Weights for each observation.
        n_samples (int): Number of MCMC samples to draw.

    Returns:
        arviz.InferenceData: The fitted model trace containing posterior distributions.
    """
    y_values_t = np.array(y_values_t)
    
    if population_weights is None:
        population_weights = np.ones_like(y_values_t)

    # Convert population_weights to a numpy array to allow boolean masking
    population_weights = np.array(population_weights)

    # Filter out invalid values
    valid_mask = np.isfinite(y_values_t)
    y_values_clean = y_values_t[valid_mask]
    weights_clean = population_weights[valid_mask]
    n_agents = len(y_values_clean)

    if n_agents < 10:
        raise ValueError("Hierarchical model requires at least 10 valid data points.")

    with pm.Model() as hierarchical_model:
        # --- Priors for Global Parameters ---
        l = pm.TruncatedNormal("l", mu=2, sigma=0.5, lower=1.1)
        
        # --- Priors for the Distribution of Agent Beliefs (x) ---
        # NON-CENTERED PARAMETERIZATION to avoid divergences
        # We model the distribution of beliefs in logit-space
        mu_logit = pm.Normal("mu_logit", mu=0, sigma=1.5)
        sigma_logit = pm.HalfNormal("sigma_logit", sigma=1.0)
        
        # --- Latent Variable: Individual Agent Beliefs ---
        # Define individual beliefs as offsets from a standard normal
        offsets = pm.Normal("offsets", mu=0, sigma=1, shape=n_agents)
        z_scaled = mu_logit + offsets * sigma_logit
        
        # Transform back to probability space [0, 1]
        x_scaled = pm.Deterministic("x_scaled", pm.math.invlogit(z_scaled))
        x = pm.Deterministic("x", x_scaled)
        
        # --- Likelihood ---
        # Correctly define the win and loss distributions
        mu_win = pm.math.log(l * x)
        mu_loss = pm.math.log(l * (1 - x) / (l - 1)) # Corrected loss expression
        obs_sigma = pm.HalfNormal("obs_sigma", 0.5)
        
        # --- Likelihood using pm.Potential for custom weighting ---
        # Calculate log-probability of data under win and loss distributions
        # Use the modern PyMC API: pm.logp(distribution, data)
        win_dist = pm.Normal.dist(mu=mu_win, sigma=obs_sigma)
        win_logp = pm.logp(win_dist, y_values_clean)

        loss_dist = pm.Normal.dist(mu=mu_loss, sigma=obs_sigma)
        loss_logp = pm.logp(loss_dist, y_values_clean)

        # Manually compute the log-likelihood of the mixture using individual beliefs x_i
        # This is the core of the hierarchical model: each agent's likelihood is weighted by its own belief
        logp_mixture = pm.math.logsumexp(
            [pm.math.log(x) + win_logp, pm.math.log(1 - x) + loss_logp],
            axis=0
        )

        # Use pm.Potential to add the weighted log-likelihood to the model
        # This is the correct way to handle population weights.
        pm.Potential("likelihood", pt.sum(logp_mixture * weights_clean))
        
        # --- Inference ---
        # We'll use pm.sample() for better accuracy, as ADVI can have biases.
        # It will be slower but more reliable.
        # approx = pm.fit(n=30000, method='advi', obj_optimizer=pm.adagrad(learning_rate=0.1))
        # idata = approx.sample(n_samples)
        idata = pm.sample(draws=n_samples, tune=4000, chains=2, target_accept=0.99)

    return idata

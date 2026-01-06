import numpy as np
from scipy import stats, optimize

import torch  


# 1. max 
def scale_max(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return torch.tensor(np.max(np.abs(x), axis=0))

# 2. mean
def scale_mean(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return torch.tensor(np.mean(np.abs(x), axis=0)) 

# 3. median
def scale_median(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return torch.tensor(np.median(np.abs(x), axis=0)) 

# 4. percentile 
def scale_percentile(x, p=99.0):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return torch.tensor(np.percentile(np.abs(x), p, axis=0), dtype=torch.float16) 

# 5. top k mean 
def scale_topk_mean(x, k=5):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return torch.tensor(np.mean(np.sort(np.abs(x), axis=0)[-2:], axis=0))

# 6. scale trimmed mean
def scale_trimmed_mean(x, trim_frac=0.35):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    n = x.shape[0] 
    k = int(trim_frac * n) 
    return torch.tensor(np.mean(np.sort(np.abs(x), axis=0)[k:n-k], axis=0))

# 7. huber estimator
def huber_estimator(x, delta=1.35, tol=1e-6, max_iter=50):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    mu = np.median(x, axis=0)   # good robust initial guess

    for _ in range(max_iter):
        r = x - mu
        w = np.where(np.abs(r) <= delta, 1, delta / np.maximum(np.abs(r)), 1e-5)
        mu_new = np.sum(w * x, axis=0) / np.sum(w, axis=0)
        if np.all(abs(mu_new - mu) < tol):
            break
        mu = mu_new
    
    # Robust scale (like std but weighted)
    r = x - mu
    scale = np.sqrt(np.mean(np.minimum(r**2, (delta**2)), axis=0))
    return torch.tensor(mu + 3 * scale)

# 8. generalized mean
def generalized_mean(x, p=2):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return torch.tensor(np.mean(np.abs(x) ** p, axis=0)) ** (1.0 / p)

# 9. log sum exp 
def logsumexp_magnitude(x, beta=30.0):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy() 
    m = np.max(np.abs(x), axis=0)
    return torch.tensor((1.0 / beta) * (m + np.log(np.sum(np.exp(beta * (x - m)), axis=0))))

# 10. high quantile gaussian
from scipy.stats import norm

def high_quantile_gaussian(x, q=0.90):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.abs(x)
    mus, sigmas = [], [] 
    for i in range(x.shape[1]):
        mu, sigma = norm.fit(x[:, i])
        mus.append(mu)
        sigmas.append(sigma)
    mus = np.array(mus)
    sigmas = np.array(sigmas)
    return torch.tensor(norm.ppf(q, loc=mus, scale=sigmas))

# 11. high quantile gpd
from scipy.stats import genpareto

def high_quantile_gpd(x, q=0.95, threshold_percentile=0.85):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy() 
    x = np.abs(x)
    n_cols = x.shape[1]
    q_estimates = np.zeros(n_cols)
    for i in range(n_cols): 
        col = x[:, i]
        threshold = np.percentile(col, threshold_percentile * 100)
        exceedances = col[col > threshold] - threshold

        c, loc, scale = genpareto.fit(exceedances, floc=0)
        
        # Probability of exceeding threshold
        p_exceed = 1 - threshold_percentile
        
        # Compute quantile in the tail
        q_exceed = genpareto.ppf((q - threshold_percentile) / p_exceed, c, loc=loc, scale=scale)
        q_estimates[i] = threshold + q_exceed
    return torch.tensor(q_estimates)

# 12. emipirical bayes
def empirical_bayes_per_channel(X):
    """
    Empirical Bayes per-channel scale.
    X: numpy array of shape [num_tokens, num_channels]
    Returns: per-channel scales (robust to noisy maxima)
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    N, C = X.shape
    
    # Step 1: Per-channel max (or you can use RMS or mean + 3*std)
    m_c = np.max(np.abs(X), axis=0)
    
    # Step 2: Estimate global statistics
    mu0 = np.mean(m_c)        # global mean across channels
    tau2 = np.var(m_c)         # between-channel variance
    sigma2_c = np.var(X, axis=0) / N  # variance of the mean (approx)
    
    # Step 3: Compute shrinkage weights
    w = sigma2_c / (sigma2_c + tau2)
    
    # Step 4: Shrink noisy per-channel maxima
    per_channel_scale = w * m_c + (1 - w) * mu0
    return per_channel_scale

# 13. EVT
from scipy.stats import genpareto

def evt_per_channel_scale(X, tail_fraction=0.05, quantile=0.999):
    """
    EVT per-channel scale estimation.
    X: [num_tokens, num_channels]
    tail_fraction: fraction of top activations to use as tail
    quantile: target extreme quantile to extrapolate to
    Returns: per-channel EVT scales
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    N, C = X.shape
    per_channel_scale = np.zeros(C)
    
    for c in range(C):
        x_c = X[:, c]
        
        # Step 1: threshold for tail
        threshold = np.percentile(x_c, 100*(1-tail_fraction))
        tail = x_c[x_c > threshold] - threshold  # exceedances
        
        if len(tail) < 2:
            # fallback if too few tail points
            per_channel_scale[c] = x_c.max()
            continue
        
        # Step 2: fit GPD
        xi, loc, beta = genpareto.fit(tail, floc=0)
        k = len(tail)
        
        # Step 3: extrapolate to desired quantile
        if xi != 0:
            q_extreme = threshold + (beta / xi) * (((1 - quantile)/(k/N))**(-xi) - 1)
        else:
            q_extreme = threshold + beta * np.log((k/N)/(1-quantile))
        
        per_channel_scale[c] = q_extreme
        
    return per_channel_scale

# 14. empirical bayes + EVT
from scipy.stats import genpareto

def robust_per_channel_scale(X, tail_fraction=0.05, quantile=0.999):
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    N, C = X.shape
    per_channel_scale = np.zeros(C)
    
    # Global mean for shrinkage
    q_extremes = np.zeros(C)
    tail_sizes = np.zeros(C)
    
    for c in range(C):
        x_c = X[:, c]
        threshold = np.percentile(x_c, 100*(1-tail_fraction))
        tail = x_c[x_c > threshold] - threshold
        tail_sizes[c] = len(tail)
        
        if len(tail) < 2:
            q_extremes[c] = x_c.max()
            continue
        
        xi, loc, beta = genpareto.fit(tail, floc=0)
        k = len(tail)
        # EVT extrapolated quantile
        q_extreme = threshold + (beta / xi) * (((1 - quantile)/(k/N))**(-xi) - 1) if xi != 0 else threshold + beta * np.log((k/N)/(1-quantile))
        q_extremes[c] = q_extreme
    
    # Empirical Bayes shrinkage across channels
    mu0 = np.mean(q_extremes)
    tau2 = np.var(q_extremes)
    sigma2 = 1.0 / np.maximum(tail_sizes, 1)  # uncertainty ~ 1/n_tail
    w = sigma2 / (sigma2 + tau2)
    
    per_channel_scale = w * q_extremes + (1-w) * mu0
    return per_channel_scale
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def iqr(a):
    return np.quantile(a, 0.75) - np.quantile(a, 0.25)


def remove_outliers_abs(sample_paths):
    a = np.abs(sample_paths).mean(1)
    q1, q3 = np.quantile(a, [0.25, 0.75])
    iqr = q3 - q1
    mask = (q1 - 1.5*iqr <= a) & (a <= q3 + 1.5*iqr)
    return sample_paths[mask]


def ref_bernoulli_posterior(n, phat):
    """ Reference posterior for a Bernoulli likelihood with a uniform prior. """
    return stats.beta(n*phat+1, n*(1-phat)+1)


def ref_gaussian_posterior(n, emp_mean):
    """ Posterior defined by the N(0, 100) prior and standard Gaussian likelihood. """
    normalization_constant = 1 / (n + (1 / 100))
    posterior_mean = emp_mean * n * normalization_constant  # mean
    posterior_variance = normalization_constant  # variance
    return stats.norm(loc=posterior_mean, scale=posterior_variance**0.5)


def remove_outliers_1d(a):
    q1, q3 = np.quantile(a, [0.25, 0.75])
    iqr = q3 - q1
    mask = (q1 - 1.5*iqr <= a) & (a <= q3 + 1.5*iqr)
    return a[mask]


def marginal_acf(paths, Kmax=5):
    """
    return the ACF-like quantity 
        ret[k] = E(x_{n+(k+1)} x_n) / E(x_n^2), 
    assuming the marginal distributions {x_n} all equal. (this is weaker than stationarity, which further look at higher-dim marginals)
    NOTE expectation is taken over both n and K
    """
    ret = []
    for i in range(1, Kmax+1):
        t = (paths[:, i:] * paths[:, :-i]).mean(1)
        t = remove_outliers_1d(t)
        ret.append(t.mean() / (1e-5+(paths[:, i:-i]**2).mean()))
    return np.asarray(ret)


def multi_acf_diff(paths, Kmax=5):
    macf = marginal_acf(paths, Kmax)
    return np.asarray([macf[i] - macf[0] for i in range(1, Kmax)])


def sample_ref_posterior_and_data(data_type, mean_param, n, N, K):
    """ sample parameter and data imputations from the reference posterior """
    if data_type in ['bernoulli', 'coin_flip']:
        posterior_params = ref_bernoulli_posterior(n, mean_param).rvs(size=(K,))
        completion = (np.random.uniform(size=(K, N)) < posterior_params[:, None]).astype('f')
    elif data_type == 'gaussian':
        posterior_params = ref_gaussian_posterior(n, mean_param).rvs(size=(K,))
        completion = np.random.normal(size=(K, N)) + posterior_params[:, None]
        # NOTE: match the discretizatione error due to LLM tokenization
        completion = (completion * 10).astype('i').astype('f') / 10
    return posterior_params, completion


def plot_approxMP(sample_paths, model_name, prompt_dataset, data_name):
    """
    Computes and visualizes the approximate MP for Bernoulli or Gaussian observations, assuming 
    the model is approximately CID
    """
    approx_MP_samples = sample_paths.astype('f').mean(1)
    MP_N = sample_paths.shape[1]
    n = prompt_dataset.shape[0]
    empirical_mean = prompt_dataset.mean()

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 5), facecolor='w', sharex=True)
    axs[0].set_title('MP approximate samples, iqr={:.2f}'.format(iqr(approx_MP_samples)))
    _ = axs[0].hist(approx_MP_samples, bins=20, density=True, alpha=0.6)
    axs[0].tick_params('x', labelbottom=False)

    def sample_Bayes_MP_estimate(p, n_samples=2000):
        """
        approx_MP_samples has an extra variance due to the use of finite (MP_N) samples in estimating
        the MP limiting parameter.
        To compare this distribution with that from a reasonable posterior (the BvM limit),
        we do the same thing to the latter: first draw theta_i ~ the BvM posterior, and then
        X_ij ~ theta_i, and \hat\theta_i = MLE(X_ij).
        """
        _, data_samples =sample_ref_posterior_and_data(data_name, p, n, MP_N, n_samples)
        # param_samples = ref_bernoulli_posterior(n, p).rvs(size=(n_samples,))
        # data_samples = (np.random.uniform(size=(n_samples, MP_N)) < param_samples[:, None])
        return data_samples.mean(1)

    ref_post_samples = sample_Bayes_MP_estimate(empirical_mean)
    axs[1].set_title('reference posterior, iqr={:.2f}'.format(iqr(ref_post_samples)))
    axs[1].hist(ref_post_samples, bins=20, density=True, alpha=0.4)
    axs[1].tick_params('x', labelbottom=False)

    mm_post_samples = sample_Bayes_MP_estimate(approx_MP_samples.mean())
    axs[2].set_title('reference posterior, centered at MP mean, iqr={:.2f}'.format(iqr(mm_post_samples)))
    axs[2].hist(mm_post_samples, bins=20, density=True,
                alpha=0.4, label='reference posterior, centered at MLE')

    plt.suptitle(f'{model_name}: comparison between approx. samples from the "MP",\nthe reference posterior based on observed data,\nand a ref. posterior with matching mean')
    plt.tight_layout()
    return fig

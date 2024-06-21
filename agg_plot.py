""" for the aggregated plots in the paper """ 

import os, os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import numpy as np
import matplotlib
import tqdm
from matplotlib import pyplot as plt

from collections import namedtuple
from typing import List, Tuple
from functools import partial

from plotting import remove_outliers_abs, sample_ref_posterior_and_data, iqr, multi_acf_diff


# Result from a single run. 
# data: (n, 1) array of observations, completion: (K, N) array of sampled completions
# NOTE: for the CID checks the original dump will contain multiple (shuffled) data; we only 
#       retain the first version. This is fine for our plots because we will only use the 
#       empirical mean, which is always the same
Dump = namedtuple('Dump', 'data completion')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=osp.expanduser('~/run/bllm-dump'))
    parser.add_argument('--save_path', type=str, default='/tmp')
    parser.add_argument('--plot', type=str, default='bernoulli-mp-check', choices=[
        'bernoulli-mp-check', 'gaussian-mp-check', 'nlp-mp-check', 'bernoulli-eu'])
    parser.add_argument('--bern_models', type=str, 
                        default=['gpt-3d', 'gpt-4', 'llama', 'mistral'], nargs='+',
                        help='models to include in the Bernoulli MP plots')
    parser.add_argument('--mpc_n', type=int, default=[50, 100], nargs='+', 
                        help='number of observations in the Bernoulli/Gaussian MP checks')
    parser.add_argument('--mpc_N_ratio', type=float, default=[0.5, 2], nargs='+',
                        help='ratio of N/n for the Bernoulli/Gaussian MP checks')
    # parser.add_argument('-d', type=str)
    return parser


def load_th_hps(hpath):
    """ load hyperparameters from a single huggingface run """
    with open(hpath) as fin:
        hps = {}
        for line in fin.readlines():
            if line.strip() == '': continue
            toks = line.strip().split(': ')
            hps[toks[0]] = ': '.join(toks[1:])
    return hps


def load_th_dump(path):
    """ load experiment dump from a single huggingface run """
    import torch as th
    data = th.load(os.path.join(path, 'obs_data_latest.th'))
    completion = th.load(os.path.join(path, 'sample_paths_latest.th'))
    return Dump(data=data.numpy()[:1, :].T,
                completion=completion.numpy())


def filter_th_paths(root_path, criteria) -> List[str]:
    ret = []
    for path in os.listdir(root_path):
        hpath = os.path.join(root_path, path, 'H_readable.txt')
        if not os.path.exists(hpath):
            continue
        hps = load_th_hps(hpath)
        if any(hps[k] != v for k, v in criteria.items()):
            continue
        ret.append(os.path.join(root_path, path))
    return ret


def compute_ag_stats(dump: Dump):
    K, N = dump.completion.shape
    noisy_MP_params = dump.completion.mean(1)  # estimated MP params; valid for Bernoulli and Gaussian data
    ret = {
        'mean_former': dump.completion[:, :N//2].mean(),
        'mean_latter': dump.completion[:, N//2:].mean(),
        'sqmean_former': (dump.completion[:, :N//2]**2).mean(),
        'sqmean_latter': (dump.completion[:, N//2:]**2).mean(),
        'noisy_MP_params': noisy_MP_params,
        'ar_diff': multi_acf_diff(dump.completion),  # [K,] array of ACF differences
    }
    return ret

    
def load_dumps(base_path_fmt, seed_range) -> List[Dump]:
    """ load a list of dumps from a base path format string and a range of seeds """
    dumps = []

    for s in seed_range:
        path = os.path.join(base_path_fmt.format(s), 'dump.npz')
        dump = np.load(path)
        dumps.append(Dump(    
            data = dump['observed_data'], # (n, 1)
            completion = dump['completion'])) # (K, N)
    
    return dumps


BSResTuple = Tuple[List[dict], List[List[dict]], List[List[dict]]]

def compute_ag_stats_and_bootstrap(
        dumps: List[Dump], 
        BS_K=300,  # number of replications
        data_type='bernoulli',
        ref_posterior_fractional=1.,  # 1 - standard posterior, otherwise - fractional
        trunc_N=-1,  # > 0: truncate completions to this length
    ) -> BSResTuple:
    """
    compute aggregated statistics from 
    1. a list of experiment dumps; 
    for each dump, also draw BS_K samples from the bootstrap sampling distribution of the 
    aggregated statistics using 
    2. a Bayesian posterior predictive distribution defined by a weakly informative prior 
    3. a Bayesian posterior predictive distribution with mean matched to the resp. experiment 

    :return: a tuple containing the three lists above
    """

    def bootstrap(param, n, N, K, data) -> List[dict]:
        bs_stats = []
        for _ in range(BS_K):
            _, ref_posterior_completion = sample_ref_posterior_and_data(data_type, param, n, N, K)
            ag_stats_i = compute_ag_stats(Dump(data=data, completion=ref_posterior_completion))
            bs_stats.append(ag_stats_i)
        return bs_stats

    all_ag_stats = []  # list of aggregated statistics from each run
    all_bs_stats = []  # list of list of bootstrap statistics, using a predictive model defined by the ref. posterior
    all_bs_mm_stats = []  # similar to above, but with the mean of the ref. posterior changed to the LLM-MP's mean
    for dump in tqdm.tqdm(dumps):
        assert dump.data.shape[1] == 1
        if trunc_N > 0:
            assert dump.completion.shape[1] >= trunc_N
            dump =  dump._replace(completion=dump.completion[:, :trunc_N])
        dump = dump._replace(completion=remove_outliers_abs(dump.completion))  # NOTE 
        n, (K, N) = dump.data.shape[0], dump.completion.shape
        all_ag_stats.append(compute_ag_stats(dump))
        emp_mean = dump.data.mean()

        bs_n_ref = int(n * ref_posterior_fractional)
        all_bs_stats.append(bootstrap(emp_mean, bs_n_ref, N, K, dump.data))

        mp_mean = dump.completion.mean()
        all_bs_mm_stats.append(bootstrap(mp_mean, bs_n_ref, N, K, dump.data))

    return all_ag_stats, all_bs_stats, all_bs_mm_stats


def mean_ag_plot(dumps: List[Dump], bs_tup: BSResTuple, test_fn='linear',
                 x_name='# trial', x_labels=None, extra_tol=-1, first=True):
    """
    aggregated plots for the first CID check 
    :param extra_tol: acceptable deviation from the CIs (-1 disables)
    """
    all_ag_stats, _, all_bs_mm_stats = bs_tup
    if test_fn == 'linear':
        key_former, key_latter ='mean_former', 'mean_latter'
    elif test_fn == 'sqr':
        key_former, key_latter = 'sqmean_former', 'sqmean_latter'
    else:
        raise ValueError(test_fn)

    Xviz = np.arange(len(dumps)) + 1
    plt.plot(Xviz, [s[key_former] - s[key_latter] for s in all_ag_stats], marker='+', label='observed')
    # 95% bootstrap CI from the ref. Bayesian model
    # NOTE: we use the mean-matched posterior. This is important for the Gaussian experiment if
    #       we want to compare the approx. posterior/MP variance without being influenced by
    #       possible differences in the mean (which will be reported separately)
    y_lo = np.array([
        np.quantile([s[key_former] - s[key_latter] for s in bs_stats], 0.025)  # for each BS sample within the experiment
        for bs_stats in all_bs_mm_stats  # for each experiment
    ])
    y_hi = np.array([
        np.quantile([s[key_former] - s[key_latter] for s in bs_stats], 0.975)
        for bs_stats in all_bs_mm_stats
    ])
    plt.fill_between(Xviz, y_lo, y_hi, color='gray', alpha=0.15, label='95% CI from Bayes. model')
    if extra_tol > 0:
        plt.fill_between(Xviz, y_lo-extra_tol, y_hi+extra_tol, color='gray', alpha=0.2)
    plt.xlabel(x_name)
    if first:
        plt.legend()
        plt.ylabel('mean(E(g(Z_{n+k}) | F_n))')
    if x_labels is not None:
        plt.xticks(Xviz, x_labels)


def ar_diff_plot(dumps: List[Dump], bs_tup: BSResTuple, 
                 x_name='# trial', x_labels=None, extra_tol=-1, first=True):
    """
    aggregated plot for the second CID check
    :param extra_tol: acceptable deviation from the CIs (-1 disables)
    """
    all_ag_stats, all_bs_stats, _ = bs_tup
    Xviz = np.arange(len(dumps)) + 1
    ar_coeffs = np.asarray([s['ar_diff'] for s in all_ag_stats])
    _ = plt.plot(Xviz, ar_coeffs[:, 0], marker='o')
    _ = plt.plot(Xviz, ar_coeffs[:, 1:], marker='o') # , linestyle=':')
    # under H0 \{ar_diff[i]\}_{i=0}^{K-1} should be identically distributed, so we use i=0
    ar_lo = [
        np.percentile([s['ar_diff'][0] for s in bs_statss], 2.5) for bs_statss in all_bs_stats]
    ar_hi = [
        np.percentile([s['ar_diff'][0] for s in bs_statss], 97.5) for bs_statss in all_bs_stats]
    ar_lo, ar_hi = map(np.asarray, (ar_lo, ar_hi))
    plt.fill_between(Xviz, ar_lo, ar_hi, alpha=0.15, color='gray',
                     label='95% CI from Bayes. model')
    if extra_tol > 0:
        plt.fill_between(Xviz, ar_lo-extra_tol, ar_hi+extra_tol, color='gray', alpha=0.2)
    # plt.title('')
    plt.xlabel(x_name)
    if first:
        plt.legend()
        plt.ylabel('AR diff')
    if x_labels is not None:
        plt.xticks(Xviz, x_labels)


MODEL_NAMES = {
    'gpt-3.5': 'GPT-3.5',
    'gpt-3d': 'GPT-3-170B',
    'gpt-3b': 'GPT-3-2.7B',
    'gpt-3b-ft': 'GPT-3-2.7B-Finetuned',
    'gpt-4': 'GPT-4',
    'llama': 'Llama-2-7B',
    'mistral': 'Mistral-7B'
}

BERN_MEAN_PARAMS = [0.3, 0.5, 0.7]
GAUSSIAN_MEAN_PARAMS = [-1, 0]


def trunc_dumps(dumps, N):
    assert all(d.completion.shape[1] >= N for d in dumps)
    return [d._replace(completion=d.completion[:, :N]) for d in dumps]


def cid_std_fig(N_OBS, N, model_prefs, root_path, save_path, data_type='bernoulli'):
    ETOL = 0.1/N_OBS
    if data_type == 'bernoulli':
        plot_funcs = [
            (r'$T_{1,g}$', 'mean', mean_ag_plot),
            (r'$T_{2,k}$', 'ard', ar_diff_plot),
        ]
        mean_params = BERN_MEAN_PARAMS
    elif data_type == 'gaussian':
        plot_funcs = [
            (r'$T_{1,g}$ for $g(z)=z$', 'mean', mean_ag_plot),
            (r'$T_{1,g}$ for $g(z)=z^2$', 'mean-sqr', partial(mean_ag_plot, test_fn='sqr')),
            (r'$T_{2,k}$', 'ard', ar_diff_plot),
        ]
        mean_params = GAUSSIAN_MEAN_PARAMS
    else:
        raise ValueError(data_type)
    # 
    dumps_by_model: List[List[Tuple[Dump, BSResTuple]]] = []
    mnames = []
    for mpref in model_prefs:
        path_template = osp.join(root_path, mpref, data_type + '-{}' + f'-n{N_OBS}-shuffled')
        dumps = load_dumps(path_template, mean_params)
        dumps = trunc_dumps(dumps, N)
        bs_tup = compute_ag_stats_and_bootstrap(dumps, BS_K=600, data_type=data_type)
        dumps_by_model.append((dumps, bs_tup))
        mnames.append(MODEL_NAMES[mpref])
    # 
    n_model = len(mnames)
    figsize = (2.75 * n_model + 0.5, 3.75)
    for suptitle_prefix, plot_name, plot_func in plot_funcs:
        plt.figure(figsize=figsize, facecolor='w')
        for i, (mname, (dumps, bs_tup)) in enumerate(zip(mnames, dumps_by_model)):
            if i==0:
                ax=plt.subplot(1, n_model, i+1)
            else:
                ax = plt.subplot(1, n_model, i+1, sharey=ax)
                ax.get_yaxis().set_visible(False)
            plot_func(
                dumps=dumps, 
                bs_tup=bs_tup,
                x_name=r'$\theta$',
                x_labels=[str(f) for f in mean_params],
                extra_tol=ETOL,
                first=False)
            plt.title(mname)
        plt.suptitle(suptitle_prefix + f', n={N_OBS}, N={N}')
        plt.tight_layout()
        path = osp.join(save_path, f'cid-{plot_name}-{data_type}-{N_OBS}-{N}.pdf')
        plt.savefig(path)
        print(f'plot saved to {path}')


def plot_scaling_main(root_path, save_path, fast=False):
    Ns = [20, 50, 100, 200, 400]
    mprefs = ['gpt-3d', 'gpt-3.5', 'llama', 'mistral']
    model_names = [MODEL_NAMES[m] for m in mprefs]
    
    dumps_by_N_and_model = []
    for model in mprefs:
        dumps_by_N = []
        for n in Ns:
            base_path = osp.join(root_path, model, f'bernoulli-0.5-n{n}-' + 's{}')
            dumps = load_dumps(base_path, range(9) if model != 'llama' else range(7))  # last few runs crashed
            dumps_by_N.append(trunc_dumps(dumps, N=n//2))
        dumps_by_N_and_model.append(dumps_by_N)

    CB = partial(compute_ag_stats_and_bootstrap, data_type='bernoulli', BS_K=500)
    CBfast = partial(compute_ag_stats_and_bootstrap, data_type='bernoulli', BS_K=10)
    if fast:
        CB = CBfast

    # for the first model, we get the MP scaling and Bayes scaling from bootstrap
    iqrs_mp, iqrs_bayes, iqrs_bayes_mm, iqr_fractional, iqr_fractional1 = [], [], [], [], []
    for dumps_by_N in dumps_by_N_and_model[0]:
        n = dumps_by_N[0].data.shape[0]
        trunc_N = n//2
        # - ag_stats & ref Bayes
        all_ag_stats, all_bs_stats, all_bs_mm_stats = CB(dumps_by_N, trunc_N=trunc_N)
        iqrs_mp.append([iqr(dct['noisy_MP_params']) for dct in all_ag_stats])
        iqrs_bayes.append([iqr(dct['noisy_MP_params']) for dct in all_bs_stats[0]])
        iqrs_bayes_mm.append([iqr(dct['noisy_MP_params']) for dct in all_bs_mm_stats[0]])
        # - fractional 0.5
        _, all_bs_frac_stats, _ = CB(dumps_by_N, ref_posterior_fractional=0.5, trunc_N=trunc_N)
        iqr_fractional.append([iqr(dct['noisy_MP_params']) for dct in all_bs_frac_stats[0]])
        # - fractional 2
        _, all_bs_frac_stats, _ = CB(dumps_by_N, ref_posterior_fractional=2., trunc_N=trunc_N)
        iqr_fractional1.append([iqr(dct['noisy_MP_params']) for dct in all_bs_frac_stats[0]])
    
    iqrs_mp, iqrs_bayes, iqrs_bayes_mm, iqr_fractional, iqr_fractional1 = map(np.asarray, (
        iqrs_mp, iqrs_bayes, iqrs_bayes_mm, iqr_fractional, iqr_fractional1))
    
    # for other models we just look at MP
    iqrs_other_mps = []
    for dumps_by_N in dumps_by_N_and_model[1:]:
        cur = []
        for dumps in dumps_by_N:
            n = dumps[0].data.shape[0]
            trunc_N = n//2
            all_ag_stats, _, _ = CBfast(dumps, trunc_N=trunc_N)
            cur.append([iqr(dct['noisy_MP_params']) for dct in all_ag_stats])
        iqrs_other_mps.append(np.asarray(cur))

    # main plot
    plt.figure(figsize=(6.5, 2.9), facecolor='w')
    lm_kw = {'marker': '+', 'linewidth': 1.25, 'alpha': 1}
    ref_kw = {'marker': '+', 'linewidth': 0.75, 'alpha': 1}
    ax = plt.subplot(111)

    for iqrs, model_name in zip([iqrs_mp] + iqrs_other_mps, model_names):
        plt.plot(Ns, np.median(iqrs, 1), label=model_name, **lm_kw)
    lns = plt.plot(Ns, np.median(iqrs_bayes, 1), label='ref. Bayesian model', 
             **(lm_kw | {'linestyle': '--', 'color': 'gray'}))
    plt.plot(Ns, np.median(iqr_fractional, 1), label=r'fractional Bayes ($\alpha$=0.5)', 
             **ref_kw, linestyle=':', color=lns[0].get_color())
    plt.plot(Ns, np.median(iqr_fractional1, 1), label=r'fractional Bayes ($\alpha$=2)', 
             **ref_kw, linestyle='-.', color=lns[0].get_color())
    def setup_plot():
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(Ns, fontsize='x-small')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.yticks([0.06, 0.1, 0.2, 0.3], fontsize='x-small')
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        _ = plt.legend(prop={'size': 12}, loc='center left', bbox_to_anchor=(1,0.5))#, fancybox=True, framealpha=0.5)
        plt.ylabel('$T_3$')
        plt.xlabel('$n$')
        plt.tight_layout()
    setup_plot()
    plt.savefig(osp.join(save_path, 'eu-scaling-all.pdf'))

    plt.figure(figsize=(7, 3), facecolor='w')
    ax = plt.subplot(111)
    for iqrs, model_name in zip([iqrs_mp] + iqrs_other_mps, model_names):
        if model_name == MODEL_NAMES['gpt-3d']:
            for s in range(iqrs.shape[1]):
                kw = {} if s>0 else {'label': 'samples of $T_3$ from GPT-3'}
                plt.scatter(Ns, iqrs[:,s], marker='x', linewidth=0.75, color='maroon', **kw)
            plt.plot(Ns, np.median(iqrs, 1), label='median($T_3$) from GPT-3', color='darkorange', marker='+', linewidth=1)
    
    iblo, ibhi = np.quantile(iqrs_bayes, [0.025, 0.975], axis=1)
    plt.fill_between(Ns, iblo, ibhi, linestyle='--', label='95\% CI from ref. Bayes. model', 
                     alpha=0.13, color='steelblue')
    setup_plot()
    plt.savefig(osp.join(save_path, 'eu-scaling-gpt3-sep.pdf'))


def synth_nlp_plot(exp_path, save_path):
    dump = np.load(os.path.join(exp_path, 'dump.npz'))

    def filter_path(arr, inp_val):
        return arr[arr[:,0]==inp_val, 1]
    
    icl_dumps = []
    icl_ns = []
    for val in [0, 1]:
        paths = []
        for i in range(dump['completion'].shape[0]):
            paths.append(filter_path(dump['completion'][i], val))
        lens = [len(p) for p in paths]
        split_len = int(np.percentile(lens, 1))
        sample_paths = np.array([p[:split_len] for p in paths if p.shape[0]>=split_len])
        relevant_observation = filter_path(dump['observed_data'], val)
        icl_ns.append(relevant_observation.shape[0])
        icl_dumps.append(Dump(data=relevant_observation[:, None],
                              completion=sample_paths[:, :20]))
        
    bs_tup = compute_ag_stats_and_bootstrap(icl_dumps, BS_K=300, data_type='bernoulli')
    n = min(icl_ns)

    plot_funcs = [
        (r'$T_{1,g}$', mean_ag_plot),
        (r'$T_{2,k}$', ar_diff_plot),
    ]

    plt.figure(figsize=(6, 3), facecolor='w')
    for i, (plot_name, plot_func) in enumerate(plot_funcs):
        plt.subplot(1, 2, 1+i)
        plot_func(
            icl_dumps, bs_tup,
            x_name='$x$', x_labels=[0, 1],
            extra_tol=0.1/n, first=False)
        plt.title(plot_name, fontsize='small')
        plt.yticks(fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f'plot saved to {save_path}')


if __name__ == '__main__':
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    font = {'family' : 'DejaVu Sans',
            'size'   : 17}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['text.latex.preamble'] = r'''\usepackage{amsmath}
    \usepackage{amssymb}'''

    args = get_parser().parse_args()
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.plot == 'bernoulli-mp-check':
        for n in args.mpc_n:
            for N_ratio in args.mpc_N_ratio:
                cid_std_fig(n, int(n*N_ratio), args.bern_models, args.root_path, args.save_path, 'bernoulli')

    elif args.plot == 'gaussian-mp-check':
        models = ['gpt-3d', 'gpt-3.5', 'llama', 'mistral']
        for n in args.mpc_n:
            for N_ratio in args.mpc_N_ratio:
                cid_std_fig(n, int(n*N_ratio), models, args.root_path, args.save_path, 'gaussian')

    elif args.plot == 'nlp-mp-check':
        models = ['gpt-3b', 'gpt-3b-ft', 'gpt-3d', 'gpt-3.5', 'gpt-4']
        for model in models:
            path = osp.join(args.root_path, model, 'synth-nlp-n80-shuffled')
            save_path = osp.join(args.save_path, 'snyth-nlp-' + model + '.pdf')
            synth_nlp_plot(path, save_path)

    elif args.plot == 'bernoulli-eu':
        plot_scaling_main(args.root_path, args.save_path)

    else:
        raise ValueError(args.plot)
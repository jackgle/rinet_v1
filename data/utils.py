import numpy as np
import scipy.optimize as opt
import scipy.stats
import scipy.special
from scipy.stats import norm, halfnorm


def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def linear(x, a, b):
    return a * x + b


def _curve_fit_nonan(func, x, y, **kwargs):
    """curve_fit with NaN values removed (compatible with newer scipy)."""
    kwargs.pop('nan_policy', None)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return opt.curve_fit(func, x[mask], y[mask], **kwargs)


def quantize_data(data, step):
    """
    Quantize the input data by rounding each float to the nearest multiple of the given step.
    """
    quantized_data = [round(x / step) * step for x in data]
    return quantized_data


def create_mixture_sample(
    params,
    sample_size=1000,
    p_frac=0.4,
    p_ratio=0.5,
    quantize_step=0
):
    """
    Sample from a generated mixture.
    """
    # get sizes
    bg_fraction = 0.001
    ref_size = int(sample_size * (1 - p_frac - bg_fraction))
    p_sizes = int(sample_size * p_frac)
    p_sizes = [int(p_sizes * p_ratio), int(p_sizes * (1 - p_ratio))]

    # reference component
    ref_model = norm(loc=2.571, scale=1.104715)
    ref_comp = ref_model.rvs(ref_size)
    ref_comp = ref_comp[ref_comp >= 0]
    if len(ref_comp) < ref_size:
        ref_comp = ref_model.rvs(ref_size * 3)
        ref_comp = ref_comp[ref_comp >= 0]
        ref_comp = ref_comp[np.random.choice(len(ref_comp), ref_size, replace=False)]
    ref_comp = scipy.special.inv_boxcox(ref_comp, params['nonp_lambda'])

    # target params
    target = []
    for i in [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]:
        target.append((scipy.special.inv_boxcox(ref_model.ppf(i), params['nonp_lambda']) - ref_comp.mean()) / ref_comp.std())
    target = np.array(target)

    ref_comp -= ref_comp.mean()
    ref_comp /= ref_comp.std()

    # left path comp
    left_comp = norm(loc=-params['left_mean_abs'], scale=params['left_std']).rvs(p_sizes[0])
    left_comp = left_comp[left_comp >= -params['left_edge_abs']]
    if len(left_comp) < p_sizes[0]:
        left_comp = norm(loc=-params['left_mean_abs'], scale=params['left_std']).rvs(p_sizes[0] * 3)
        while len(left_comp[left_comp >= -params['left_edge_abs']]) < p_sizes[0] / 10:
            left_comp = norm(loc=-params['left_mean_abs'], scale=params['left_std']).rvs(p_sizes[0] * 3)
        left_comp = left_comp[left_comp >= -params['left_edge_abs']]
        if len(left_comp) < p_sizes[0]:
            left_comp = left_comp[np.random.choice(len(left_comp), p_sizes[0], replace=True)]
        else:
            left_comp = left_comp[np.random.choice(len(left_comp), p_sizes[0], replace=False)]

    # right path comp
    right_comp = norm(loc=params['right_mean'], scale=params['right_std']).rvs(p_sizes[1])
    if len(right_comp) < p_sizes[1]:
        right_comp = norm(loc=-params['right_mean_abs'], scale=params['right_std']).rvs(p_sizes[1] * 3)
        right_comp = right_comp[right_comp >= -params['left_edge_abs']]
        right_comp = right_comp[np.random.choice(len(right_comp), p_sizes[1], replace=0)]

    # background noise
    bg_noise = np.random.uniform(params['left_edge_abs'], params['bg_max'], int(sample_size * bg_fraction))

    mixture = np.hstack([ref_comp, left_comp, right_comp, bg_noise])

    if quantize_step:
        mixture = quantize_data(mixture, quantize_step)

    comp_sizes = np.array([len(ref_comp), len(left_comp), len(right_comp)])

    return mixture, target, comp_sizes


class RIbenchModeler:
    def __init__(self, bin_shift, max_skew):
        self.bin_shift = bin_shift
        self.max_skew = max_skew
        self.bin_starts = np.arange(0, max_skew, bin_shift)
        self.min_std = 0.1
        self.min_mean_abs = 0
        self.fits = {}
        self.data = {}

    def fit(self, ref_skew_sorted, left_mean_abs, right_mean, left_std, right_std,
            left_edge_abs, bg_max, plot=False):
        """
        Model the parameters of RIbench mixtures as functions of the reference component skew.

        All arrays should be sorted by ref_skew.
        """
        bin_starts = self.bin_starts
        max_skew = self.max_skew

        x_data = ref_skew_sorted[ref_skew_sorted <= max_skew]
        self.data['x_data'] = x_data

        # left pathological means
        bin_width = 3
        self.data['left_mean_abs'] = left_mean_abs[ref_skew_sorted <= max_skew]
        y_data = self.data['left_mean_abs']
        means = np.array([np.mean(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popm, _ = _curve_fit_nonan(exp_decay, bin_starts + bin_width / 2, means)
        stds = np.array([np.std(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popv, _ = _curve_fit_nonan(exp_decay, bin_starts + bin_width / 2, stds)
        self.fits['left_mean_abs'] = {
            'mean': {'func': exp_decay, 'params': popm},
            'std': {'func': exp_decay, 'params': popv}
        }

        # right pathological means
        bin_width = 3
        self.data['right_mean'] = right_mean[ref_skew_sorted <= max_skew]
        y_data = self.data['right_mean']
        means = np.array([np.mean(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popm, _ = _curve_fit_nonan(linear, bin_starts + bin_width / 2, means)
        stds = np.array([np.std(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popv, _ = _curve_fit_nonan(linear, bin_starts + bin_width / 2, stds)
        self.fits['right_mean'] = {
            'mean': {'func': linear, 'params': popm},
            'std': {'func': linear, 'params': popv}
        }

        # left stds
        bin_width = 3
        self.data['left_std'] = left_std[ref_skew_sorted <= max_skew]
        y_data = self.data['left_std']
        means = np.array([np.mean(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popm, _ = _curve_fit_nonan(exp_decay, bin_starts + bin_width / 2, means)
        stds = np.array([np.std(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popv, _ = _curve_fit_nonan(exp_decay, bin_starts + bin_width / 2, stds)
        self.fits['left_std'] = {
            'mean': {'func': exp_decay, 'params': popm},
            'std': {'func': exp_decay, 'params': popv}
        }

        # right stds
        bin_width = 3
        self.data['right_std'] = right_std[ref_skew_sorted <= max_skew]
        y_data = self.data['right_std']
        means = np.array([np.mean(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popm, _ = _curve_fit_nonan(exp_decay, bin_starts + bin_width / 2, means)
        stds = np.array([np.std(y_data[(x_data >= i) & (x_data < (i + bin_width))]) for i in bin_starts])
        popv, _ = _curve_fit_nonan(exp_decay, bin_starts + bin_width / 2, stds)
        self.fits['right_std'] = {
            'mean': {'func': exp_decay, 'params': popm},
            'std': {'func': exp_decay, 'params': popv}
        }

        # left_edge (same as background minimum)
        self.data['left_edge_abs'] = left_edge_abs[ref_skew_sorted <= max_skew]
        y_data = self.data['left_edge_abs']
        popm, _ = opt.curve_fit(exp_decay, x_data[x_data >= 0.1], y_data[x_data >= 0.1])
        self.fits['left_edge_abs'] = {
            'mean': {'func': exp_decay, 'params': popm},
            'std': {'func': linear, 'params': [-50, 9]}
        }

        # bg_max
        self.data['bg_max'] = bg_max[ref_skew_sorted <= max_skew]
        y_data = self.data['bg_max']
        popm, _ = opt.curve_fit(exp_decay, x_data[x_data >= 0.1], y_data[x_data >= 0.1])
        self.fits['bg_max'] = {
            'mean': {'func': exp_decay, 'params': popm},
            'std': {'func': linear, 'params': [-50 * 10, 80]}
        }

        # nonp lambda
        temp_lambda = []
        temp_skew = []
        repeats = 10
        self.data['nonp_lambda'] = {'x': [], 'y': []}
        for i in np.linspace(1, 0, 20):
            for repeat in range(repeats):
                temp_data = np.random.normal(2.571, 1.104715, 10000)
                temp_data = scipy.special.inv_boxcox(temp_data[temp_data >= 0], i)
                temp_data -= temp_data.mean()
                temp_data /= temp_data.std()
                temp_lambda.append(i)
                temp_skew.append(scipy.stats.skew(temp_data))
                self.data['nonp_lambda']['x'].append(temp_skew[-1])
                self.data['nonp_lambda']['y'].append(temp_lambda[-1])
        temp_lambda = np.array(temp_lambda)
        temp_skew = np.array(temp_skew)
        bin_width = 1
        means = np.array([np.mean(temp_lambda[(temp_skew >= i) & (temp_skew < (i + bin_width))]) for i in bin_starts])
        popm, _ = _curve_fit_nonan(exp_decay, bin_starts + bin_width / 2, means)
        self.fits['nonp_lambda'] = {
            'mean': {'func': exp_decay, 'params': popm},
            'std': {'func': linear, 'params': [0, 0]}
        }

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.subplot(4, 2, 1)
            self.plot_fit('left_mean_abs')
            plt.subplot(4, 2, 2)
            self.plot_fit('right_mean')
            plt.subplot(4, 2, 3)
            self.plot_fit('left_std')
            plt.subplot(4, 2, 4)
            self.plot_fit('right_std')
            plt.subplot(4, 2, 5)
            self.plot_fit('left_edge_abs')
            plt.subplot(4, 2, 6)
            self.plot_fit('bg_max')
            plt.subplot(4, 2, 7)
            self.plot_fit('nonp_lambda')
            plt.tight_layout()

    def plot_fit(self, param_name):
        """Plot fitted parameter functions."""
        import matplotlib.pyplot as plt
        x = np.linspace(0, 10, 200)
        if param_name != 'nonp_lambda':
            plt.scatter(self.data['x_data'], self.data[param_name], alpha=0.25, s=10)
        else:
            plt.scatter(self.data[param_name]['x'], self.data[param_name]['y'], alpha=0.25, s=10)
        plt.plot(
            x,
            np.maximum(
                self.min_mean_abs,
                self.fits[param_name]['mean']['func'](x, *self.fits[param_name]['mean']['params'])
            ),
            c='r', label='mean'
        )
        if param_name not in ['left_edge_abs', 'bg_max']:
            plt.plot(
                x,
                [
                    np.maximum(self.min_std, norm(
                        self.fits[param_name]['mean']['func'](i, *self.fits[param_name]['mean']['params']),
                        self.fits[param_name]['std']['func'](i, *self.fits[param_name]['std']['params'])).ppf(0.025))
                    for i in x
                ], c='k', linestyle=':'
            )
        plt.plot(
            x,
            [
                np.maximum(self.min_std, norm(
                    self.fits[param_name]['mean']['func'](i, *self.fits[param_name]['mean']['params']),
                    self.fits[param_name]['std']['func'](i, *self.fits[param_name]['std']['params'])).ppf(0.975))
                for i in x
            ], c='k', linestyle=':', label='95% confidence'
        )
        plt.title(param_name)
        plt.legend()

    def generate(self, ref_skew):
        """Generate pathological component parameters based on the fits and a reference skew."""
        params = {}
        for key in self.fits.keys():
            if key != 'nonp_lambda':
                if key not in ['left_edge_abs', 'bg_max']:
                    model = norm(
                        loc=self.fits[key]['mean']['func'](ref_skew, *self.fits[key]['mean']['params']),
                        scale=np.maximum(0, self.fits[key]['std']['func'](ref_skew, *self.fits[key]['std']['params']))
                    )
                    params[key] = model.rvs(1)
                else:
                    model = halfnorm(
                        loc=self.fits[key]['mean']['func'](ref_skew, *self.fits[key]['mean']['params']),
                        scale=np.maximum(0, self.fits[key]['std']['func'](ref_skew, *self.fits[key]['std']['params']))
                    )
                    params[key] = model.rvs(1)

                if 'mean' in key:
                    params[key] = np.maximum(self.min_mean_abs, params[key])
                if 'std' in key:
                    params[key] = np.maximum(self.min_std, params[key])
            else:
                params[key] = self.fits[key]['mean']['func'](ref_skew, *self.fits[key]['mean']['params'])
        # make sure left component is not too low based on the left edge
        params['left_mean_abs'] = -np.maximum(-params['left_edge_abs'] - params['left_std'], -params['left_mean_abs'])
        return params


def load_ribench_params(csv_path, exclude_analytes=('CRP', 'LDH'), fraction_pathol=0.50):
    """Load and filter RIbench parameters, returning unique component parameter sets."""
    import pandas as pd
    params = pd.read_csv(csv_path, index_col=0)
    for analyte in exclude_analytes:
        params = params[params.Analyte != analyte]
    params = params[params.fractionPathol == fraction_pathol]
    subset_cols = ['nonp_mu', 'nonp_sigma', 'nonp_lambda', 'left_mu', 'left_sigma',
                   'right_mu', 'right_sigma', 'bg_max', 'bg_min']
    params = params.drop_duplicates(subset=subset_cols)
    params = params.reset_index(drop=True)
    return params


def compute_ref_stats(params):
    """
    Compute reference component statistics from RIbench params.

    Returns dict with: ref_mean, ref_std, ref_zero, ref_skew, ref_pdf, analytes, indices
    """
    from scipy.special import inv_boxcox

    ref_mean, ref_std, ref_pdf, ref_zero, ref_skew = [], [], [], [], []
    x = np.linspace(-10, 10, 200)

    for c, i in enumerate(params.iterrows()):
        i = i[1]
        data = np.random.normal(i['nonp_mu'], i['nonp_sigma'], 50000)
        data = inv_boxcox(data, i['nonp_lambda'])
        ref_mean.append(data.mean())
        ref_std.append(data.std())
        ref_zero.append(-ref_mean[-1] / ref_std[-1])
        data = (data - data.mean()) / data.std()
        ref_skew.append(scipy.stats.skew(data))
        kde = scipy.stats.gaussian_kde(data)
        ref_pdf.append(kde(x))

    return {
        'ref_mean': np.array(ref_mean),
        'ref_std': np.array(ref_std),
        'ref_zero': np.array(ref_zero),
        'ref_skew': np.array(ref_skew),
        'ref_pdf': np.array(ref_pdf),
        'analytes': np.array(params.Analyte),
        'indices': np.array(params.Index),
    }


def compute_pathological_stats(params, ref_mean, ref_std):
    """
    Compute standardized pathological statistics from RIbench params.

    Returns p_mean, p_std arrays.
    """
    p_mean, p_std = [], []
    for c, i in enumerate(params.iterrows()):
        i = i[1]
        p_mean.append([
            (i['left_mu'] - ref_mean[c]) / ref_std[c],
            (i['right_mu'] - ref_mean[c]) / ref_std[c]
        ])
        p_std.append([
            i['left_sigma'] / ref_std[c],
            i['right_sigma'] / ref_std[c]
        ])
    return np.array(p_mean), np.array(p_std)


def collect_sorted_stats(ref_skew, p_mean, p_std, ref_zero, bg_range_data=None):
    """
    Collect and sort pathological statistics by ref_skew.

    bg_range_data should be list of [bg_min_std, bg_max_std] pairs, or None if bg_max
    is computed separately.

    Returns dict with sorted arrays keyed by stat name, plus ref_skew_sorted.
    """
    sort_idx = np.argsort(ref_skew)
    result = {
        'left_mean_abs': np.abs([i[0] for i in p_mean])[sort_idx],
        'right_mean': np.array([i[1] for i in p_mean])[sort_idx],
        'left_std': np.array([i[0] for i in p_std])[sort_idx],
        'right_std': np.array([i[1] for i in p_std])[sort_idx],
        'left_edge_abs': np.abs(ref_zero)[sort_idx],
        'ref_skew_sorted': np.sort(np.array(ref_skew)),
    }
    if bg_range_data is not None:
        result['bg_max'] = np.array([i[1] for i in bg_range_data])[sort_idx]
    return result

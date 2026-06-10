# equibin

[![CI](https://github.com/det-lab/equibin/actions/workflows/ci.yml/badge.svg)](https://github.com/det-lab/equibin/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/equibin)](https://pypi.org/project/equibin/)

2D equal-probability binning for statistical data analysis.

Partitions a 2D dataset into bins that each contain approximately the same
number of points. At each step the dimension with the highest variance is split
at its median, recursively, until the target number of bins is reached. The
resulting bins are axis-aligned rectangles that adapt to the local density of
the data.

This implements the multivariate probability binning algorithm described in:

> Roederer, M., Moore, W., Treister, A., Hardy, R. R. & Herzenberg, L. A. (2001).
> Probability binning comparison: a metric for quantitating multivariate distribution differences.
> *Cytometry* 45(1):47–55.
> [https://doi.org/10.1002/1097-0320(20010901)45:1<47::AID-CYTO1143>3.0.CO;2-A](https://doi.org/10.1002/1097-0320%2820010901%2945:1%3C47::AID-CYTO1143%3E3.0.CO;2-A)

## Installation

```
pip install equibin
```

## Usage

```python
import numpy as np
from equibin import bin_2d, plot_bins, save_bins

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 5000)
y = rng.uniform(0, 10, 5000)

result = bin_2d(x, y, n_bins=128)

print(len(result))          # 128
print(result.counts.sum())  # 5000
print(result.bins[0])       # (xmin, xmax, ymin, ymax)
```

Restrict binning to a region of interest:

```python
result = bin_2d(x, y, n_bins=256, xmin=2.5, xmax=20, ymin=0.25, ymax=10)
```

Plot the bins overlaid on the data:

```python
plot_bins(result, x, y, title="Equal-probability bins", xlim=(0, 10), ylim=(0, 10))
```

Save bin boundaries to a text file (one line per bin: `xlo xhi ylo yhi label`):

```python
save_bins(result, "bins.txt", label_prefix="run_")
```

## Authors

- Amy Roberts (amy.roberts@ucdenver.edu)
- Lekhraj Pandey (lekhraj.pandey@coyotes.usd.edu)
- Anthony Villano (anthony.villano@ucdenver.edu)

## License

GPL-2.0-only

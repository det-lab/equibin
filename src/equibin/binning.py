"""
2D equal-probability binning (multivariate probability binning).

Algorithm from:
    Roederer, M., Moore, W., Treister, A., Hardy, R. R. & Herzenberg, L. A. (2001).
    Probability binning comparison: a metric for quantitating multivariate distribution
    differences. Cytometry 45(1):47-55.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Bin = tuple[float, float, float, float]


@dataclass
class BinningResult:
    """
    Result of 2D equal-probability binning.

    `counts[i]` is the data count for `bins[i]`.
    `bins[i]` is `(xmin, xmax, ymin, ymax)`.
    """

    counts: npt.NDArray[np.intp]
    bins: list[Bin]

    def __len__(self) -> int:
        return len(self.bins)


def bin_2d(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    n_bins: int = 128,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
) -> BinningResult:
    """
    Partition 2D data into `n_bins` equal-count bins using multivariate
    probability binning.

    At each recursive step the dimension with the highest variance is split at
    its median until `n_bins` bins are produced. `n_bins` need not be a power
    of two. Algorithm from:

        Roederer, M., Moore, W., Treister, A., Hardy, R. R. & Herzenberg, L. A. (2001).
        Probability binning comparison: a metric for quantitating multivariate distribution
        differences. Cytometry 45(1):47-55.

    Parameters
    ----------
    x, y : array-like
        1-D coordinate arrays of equal length.
    n_bins : int
        Target number of output bins.
    xmin, xmax, ymin, ymax : float, optional
        Points outside these bounds are excluded before binning.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)

    mask = np.ones(len(xa), dtype=bool)
    if xmin is not None:
        mask &= xa >= xmin
    if xmax is not None:
        mask &= xa <= xmax
    if ymin is not None:
        mask &= ya >= ymin
    if ymax is not None:
        mask &= ya <= ymax

    data = np.column_stack((xa[mask], ya[mask]))

    def _split(
        d: npt.NDArray[np.float64],
        n: int,
        bounds: list[tuple[float, float]],
    ) -> list[tuple[list[tuple[float, float]], int]]:
        if n == 1 or len(d) == 0:
            return [(list(bounds), len(d))]
        split_dim = int(np.argmax(np.var(d, axis=0)))
        median = float(np.median(d[:, split_dim]))
        left_mask = d[:, split_dim] <= median
        left_bounds = list(bounds)
        right_bounds = list(bounds)
        left_bounds[split_dim] = (bounds[split_dim][0], median)
        right_bounds[split_dim] = (median, bounds[split_dim][1])
        return _split(d[left_mask], n // 2, left_bounds) + _split(
            d[~left_mask], n - n // 2, right_bounds
        )

    if len(data) == 0:
        init: list[tuple[float, float]] = [
            (xmin if xmin is not None else 0.0, xmax if xmax is not None else 1.0),
            (ymin if ymin is not None else 0.0, ymax if ymax is not None else 1.0),
        ]
    else:
        init = [
            (float(data[:, 0].min()), float(data[:, 0].max())),
            (float(data[:, 1].min()), float(data[:, 1].max())),
        ]

    raw = _split(data, n_bins, init)
    counts = np.array([r[1] for r in raw], dtype=np.intp)
    bins: list[Bin] = [
        (float(r[0][0][0]), float(r[0][0][1]), float(r[0][1][0]), float(r[0][1][1])) for r in raw
    ]
    return BinningResult(counts=counts, bins=bins)


def plot_bins(
    result: BinningResult,
    x: npt.ArrayLike | None = None,
    y: npt.ArrayLike | None = None,
    *,
    title: str | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plot bin rectangles, optionally overlaid on a scatter of the source data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _fig, ax = plt.subplots(figsize=(8, 8))

    if x is not None and y is not None:
        ax.scatter(np.asarray(x), np.asarray(y), s=5, alpha=0.5)

    for xlo, xhi, ylo, yhi in result.bins:
        ax.add_patch(
            Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, edgecolor="red", facecolor="none", lw=1)
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.show()


def save_bins(
    result: BinningResult,
    output_file: str | Path,
    label_prefix: str = "",
) -> None:
    """
    Write bin boundaries to a whitespace-delimited text file.

    Each line: ``xlo xhi ylo yhi <label_prefix><index>``.
    """
    path = Path(output_file)
    lines = [
        f"{xlo} {xhi} {ylo} {yhi} {label_prefix}{i}"
        for i, (xlo, xhi, ylo, yhi) in enumerate(result.bins)
    ]
    path.write_text("\n".join(lines) + "\n")


## Tests


def test_bin_2d_returns_correct_count() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 1000)
    y = rng.uniform(0, 10, 1000)
    result = bin_2d(x, y, n_bins=32)
    assert len(result) == 32
    assert result.counts.sum() == 1000


def test_bin_2d_bounds_filtering() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 2000)
    y = rng.uniform(0, 10, 2000)
    full = bin_2d(x, y, n_bins=16)
    bounded = bin_2d(x, y, n_bins=16, xmin=2, xmax=8, ymin=2, ymax=8)
    assert bounded.counts.sum() < full.counts.sum()


def test_bin_2d_roughly_equal_counts() -> None:
    rng = np.random.default_rng(2)
    n = 10_000
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    result = bin_2d(x, y, n_bins=128)
    expected = n / 128
    # Uniform data should give counts within 30% of expected
    assert (np.abs(result.counts - expected) < 0.3 * expected).all()


def test_save_bins() -> None:
    rng = np.random.default_rng(3)
    result = bin_2d(rng.uniform(0, 1, 100), rng.uniform(0, 1, 100), n_bins=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "bins.txt"
        save_bins(result, out, label_prefix="run_")
        lines = out.read_text().splitlines()
    assert len(lines) == 4
    assert lines[0].endswith("run_0")
    assert lines[3].endswith("run_3")


def test_plot_bins_runs() -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 200)
    y = rng.uniform(0, 1, 200)
    result = bin_2d(x, y, n_bins=8)
    plot_bins(result, x, y, title="Test", xlim=(0, 1), ylim=(0, 1))
    plt.close("all")

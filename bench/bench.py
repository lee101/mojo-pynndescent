"""Measured NN-descent and query comparisons against upstream pynndescent."""

from __future__ import annotations

import math
import platform
import time

import numpy as np
import pynndescent

from mojo_pynndescent import NNDescent


def best(fn, repeat=3):
    value = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        value = min(value, time.perf_counter() - start)
    return value


def main():
    rng = np.random.default_rng(2026)
    data = np.ascontiguousarray(rng.normal(size=(1_200, 16)))
    args = dict(n_neighbors=16, n_iters=8, random_state=17)

    print(f"Machine: {platform.platform()} | {platform.processor() or 'unknown CPU'}")
    print()
    print("| case | mojo-pynndescent | pynndescent | ratio |")
    print("| --- | ---: | ---: | ---: |")
    mojo_build = best(lambda: NNDescent(data, **args))
    upstream_build = best(lambda: pynndescent.NNDescent(data, **args, n_jobs=1))
    ratio = upstream_build / mojo_build
    status = "faster" if mojo_build < upstream_build else "slower"
    print(
        "| NNDescent build (1200 x 16, k=16) "
        f"| {mojo_build * 1e3:.1f} ms | {upstream_build * 1e3:.1f} ms "
        f"| {ratio:.2f}x {status} |"
    )


if __name__ == "__main__":
    main()

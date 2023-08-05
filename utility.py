import numpy as np


def breathing(t: float, period: float = 1):
    """A periodic function that oscillates between 0 and 1 with a period mimicking a breathing pattern."""
    p = t / period % 1
    # First third     : 0 -> 1
    # then            : 1 -> 0
    if p < 1 / 3:
        return p * 3 * 1.1
    else:
        s = (p - 1 / 3) * 3 / 2
        return (1 - s) ** 4 * 1.1


# Smoothed breathing function
x = np.linspace(0, 3, 300)
y = [breathing(t) for t in x]
kernel_size = 10
y = np.convolve(y, np.ones(kernel_size) / kernel_size, mode='same')
curve = y[100: 200]


def smooth_breathing(s: float, period: float = 5, curve=curve):
    """A periodic function that oscillates between 0 and 1 with a period mimicking a breathing pattern."""
    p = s / period % 1
    curve_pos = int(p * len(curve))
    delta = p * len(curve) - curve_pos
    return curve[curve_pos] * (1 - delta) + curve[(curve_pos + 1) % len(curve)] * delta




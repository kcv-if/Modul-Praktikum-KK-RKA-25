"""Scalar membership functions introduced in fuzzy notebook 01.

Every function returns a float in [0, 1]. The shoulder functions saturate
outside their transition interval; the other functions have finite support.
"""


def linear_up(x, a, b):
    """Return a rising linear shoulder with transition from a to b."""
    if not a < b:
        raise ValueError("Expected a < b")
    return float(max(0.0, min(1.0, (x - a) / (b - a))))


def linear_down(x, a, b):
    """Return a falling linear shoulder with transition from a to b."""
    if not a < b:
        raise ValueError("Expected a < b")
    return float(max(0.0, min(1.0, (b - x) / (b - a))))


def triangular(x, a, b, c):
    """Return a triangular membership degree with peak at b."""
    if not a < b < c:
        raise ValueError("Expected a < b < c")
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return float((x - a) / (b - a))
    return float((c - x) / (c - b))


def trapezoidal(x, a, b, c, d):
    """Return a trapezoidal membership degree with plateau [b, c]."""
    if not a < b <= c < d:
        raise ValueError("Expected a < b <= c < d")
    if x <= a or x >= d:
        return 0.0
    if x <= b:
        return float((x - a) / (b - a))
    if x <= c:
        return 1.0
    return float((d - x) / (d - c))


def sigmoid(x, a, b, c):
    """Return the piecewise S-curve from slide 11 (b is the midpoint)."""
    if not a < b < c or abs(2 * b - a - c) > 1e-9:
        raise ValueError("Expected a < b < c and b = (a + c) / 2")
    if x <= a:
        return 0.0
    if x <= b:
        return float(2 * ((x - a) / (c - a)) ** 2)
    if x < c:
        return float(1 - 2 * ((c - x) / (c - a)) ** 2)
    return 1.0


def phi(x, a, b, c):
    """Return a bell curve on [a, c] with peak b using two S-curves."""
    if not a < b < c:
        raise ValueError("Expected a < b < c")
    if x <= b:
        return sigmoid(x, a, (a + b) / 2, b)
    return float(1 - sigmoid(x, b, (b + c) / 2, c))

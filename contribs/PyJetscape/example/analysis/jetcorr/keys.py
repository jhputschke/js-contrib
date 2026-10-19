"""Angles and portable random choices: the same numbers in Python and C++
(``analysis_root/common/AnaCommon.h``).  A 64-bit key of (seed, the production file's name,
the event in the file) drives every random choice of the analyses (oversamples, partners,
random cones), so a choice does not depend on which other files a campaign holds."""
import numpy as np


def wrap_dphi(dphi, lo=-np.pi / 2):
    """Angles into [lo, lo + 2 pi)."""
    return np.mod(np.asarray(dphi) - lo, 2 * np.pi) + lo


# ── portable random choices (the same in analysis_root/common/AnaCommon.h) ────────────
_M64 = (1 << 64) - 1


def splitmix64(x):
    """SplitMix64's output function of a 64-bit integer."""
    x = (int(x) + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def fnv1a64(text):
    """FNV-1a 64-bit hash of a string's UTF-8 bytes."""
    h = 0xCBF29CE484222325
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & _M64
    return h


def event_key(seed, name, event):
    """A 64-bit key of (seed, the production file's name, the event in the file): the same
    whatever other files a campaign holds, and in C++ and Python alike."""
    return splitmix64(splitmix64(fnv1a64(name) ^ (int(seed) & _M64)) ^ (int(event) & _M64))


def key_index(key, n, salt=0):
    """An index in [0, n) from a key (and a salt for several choices per key)."""
    return splitmix64(key ^ (int(salt) & _M64)) % int(n)


def key_uniform(key, salt=0):
    """A uniform number in [0, 1) from a key and a salt (53 bits)."""
    return (splitmix64(key ^ (int(salt) & _M64)) >> 11) * 2.0 ** -53

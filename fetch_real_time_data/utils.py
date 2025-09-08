"""
Utility functions for working with time.

This module provides helpers to obtain the current wall-clock timestamp
(in milliseconds) and a monotonic clock suitable for measuring intervals.
"""

import time


def get_timestamp_ms() -> int:
    """
    Get the current wall-clock timestamp in milliseconds.

    This uses 'time.time()' under the hood, which reflects the system
    clock and may be adjusted forwards or backwards (e.g., by NTP).

    Returns
    -------
    int
        The current Unix timestamp in milliseconds (since Jan 1, 1970 UTC).
    """
    return int(time.time() * 1000)


def get_monotonic_s() -> float:
    """
    Get the current value of a monotonic clock in seconds.

    Unlike 'time.time()', this clock cannot go backwards, making it
    suitable for measuring elapsed time or implementing timeouts.
    The actual value is arbitrary (platform-dependent), but differences
    between calls are meaningful.

    Returns
    -------
    float
        Monotonic clock reading in seconds.
    """
    return time.monotonic()
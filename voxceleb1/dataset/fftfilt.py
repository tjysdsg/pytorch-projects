import numpy as np

"""
Filter data with an FIR filter using the overlap-add method.
from http://projects.scipy.org/scipy/attachment/ticket/837/fftfilt.py
"""


def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""
    return np.ceil(np.log2(np.abs(x)))


def fftfilt(b, x, n=None):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""

    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if n is not None:
        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        if type(n) != int or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2 ** nextpow2(n)
    else:
        if N_x > N_b:
            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2 ** np.arange(np.ceil(np.log2(N_b)),
                               np.floor(np.log2(N_x)) + 1)
            cost = np.ceil(N_x / (N - N_b + 1)) * N * (np.log2(N) + 1)
            N_fft = N[np.argmin(cost)]
        else:
            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2 ** nextpow2(N_b + N_x - 1)

    N_fft = int(N_fft)

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    H = np.fft.fft(b, N_fft)

    y = np.zeros(N_x, dtype=np.float32)
    i = 0
    while i <= N_x:
        il = min([i + L, N_x])
        k = min([i + N_fft, N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il], N_fft) * H, N_fft)  # Overlap..
        y[i:k] = y[i:k] + np.real(yt[:k - i])  # and add
        i += L
    return y

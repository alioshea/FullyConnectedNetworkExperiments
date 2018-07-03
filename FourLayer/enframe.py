import numpy as np
from scipy import fix, signal

def enframe(x, win, inc):
    
    nx = len(x)
    try:
        nwin = len(win)
    except TypeError:
        nwin = 1
    if nwin == 1:
        length = win
    else:
        length = nwin

    nf = int(fix((nx - length + inc) // inc))
    indf = inc * np.arange(nf)
    inds = np.arange(length) + 1
    f = x[(np.transpose(np.vstack([indf] * length)) +
           np.vstack([inds] * nf)) - 1]
    if (nwin > 1):
        w = np.transpose(win)
        f = f * np.vstack([w] * nf)
    #f = signal.detrend(f, type='constant')
    #no_win, _ = f
    return f

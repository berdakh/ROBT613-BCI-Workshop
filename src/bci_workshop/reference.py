"""Reference utilities for independent checks and student extensions."""
import numpy as np
from scipy.signal import sosfilt

def decode_row_column(scores):
    """scores: character × repetition × 12 flash groups (6 rows, then columns)."""
    scores=np.asarray(scores)
    if scores.ndim!=3 or scores.shape[-1]!=12 or scores.shape[1]<1:
        raise ValueError('Expected characters × repetitions × 12 flash groups')
    total=scores.sum(axis=1)
    return total[:,:6].argmax(axis=1)*6+total[:,6:].argmax(axis=1)

def causal_chunks(sos,samples,chunk_size):
    """Filter a 1D stream with preserved state; same zero initialization as sosfilt."""
    if chunk_size<1: raise ValueError('chunk_size must be positive')
    state=np.zeros((len(sos),2)); chunks=[]
    for start in range(0,len(samples),chunk_size):
        output,state=sosfilt(sos,samples[start:start+chunk_size],zi=state)
        chunks.append(output)
    return np.concatenate(chunks) if chunks else np.array([])

def require_disjoint_groups(train_groups,test_groups):
    """Fail if an experimental unit occurs on both sides of the split."""
    overlap=set(train_groups)&set(test_groups)
    if overlap: raise ValueError(f'Leaking groups: {sorted(overlap)}')

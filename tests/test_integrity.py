import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from scipy.signal import butter,sosfilt
from bci_workshop.reference import decode_row_column,causal_chunks,require_disjoint_groups

def test_all_symbols_and_repetition_aggregation():
    scores=np.zeros((36,3,12))
    for symbol in range(36):
        row,col=divmod(symbol,6)
        scores[symbol,:,row]=1
        scores[symbol,:,6+col]=1
    np.testing.assert_array_equal(decode_row_column(scores),np.arange(36))

@pytest.mark.parametrize('chunk_size',[1,31,128,10000])
def test_chunk_boundaries_preserve_causal_signal(chunk_size):
    x=np.random.default_rng(613).normal(size=1025)
    sos=butter(4,[8,30],fs=128,btype='bandpass',output='sos')
    np.testing.assert_allclose(causal_chunks(sos,x,chunk_size),sosfilt(sos,x),atol=1e-12)

def test_overlap_is_rejected():
    with pytest.raises(ValueError,match='Leaking'):
        require_disjoint_groups(['run1','run2'],['run2'])
    require_disjoint_groups(['run1'],['run2'])

def test_invalid_speller_schema_is_rejected():
    with pytest.raises(ValueError): decode_row_column(np.zeros((2,12)))

from process import create_label
import numpy as np
import pytest

def test_create_label_standard():
    filename = "chord_0XXXXX"
    label = np.concatenate(([1],np.zeros(36)))
    assert np.array_equal(label, create_label(filename))
    
    filename = "chord_000000"
    label = np.zeros(37)
    indexes = [0, 5, 10, 15, 19, 24]
    label[indexes] = 1
    assert np.array_equal(label, create_label(filename))

@pytest.mark.parametrize("filename", ["chord_XXXXX", "wrong", ""])
def test_create_label_invalid(filename):
    with pytest.raises(ValueError):
        create_label(filename)
    
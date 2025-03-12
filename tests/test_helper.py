from helper import find_notes, make_abs
import pytest
import os

def test_find_notes_standard():
    assert find_notes("X32010") == "C3 E3 G3 C4 E4"
    assert find_notes("000000") == "E2 A2 D3 G3 B3 E4"
    assert find_notes("XX0232") == "D3 A3 D4 F#4"

def test_find_notes_long():
    assert find_notes("10 X 12 X 10 10") == "D3 D4 A4 D5"
    assert find_notes("9 11 11 9 9 X") == "C#3 G#3 C#4 E4 G#4"
    assert find_notes("12 12 12 12 12 12") == "E3 A3 D4 G4 B4 E5"

@pytest.mark.parametrize("invalid", ["X3201", "X 10 X 12 X", "X 10 Y 12 X 10"])
def test_find_notes_invalid(invalid):
    with pytest.raises(ValueError):
        find_notes(invalid)

def test_make_abs_rel():
    rel_path = "test/hello.py"
    path = make_abs(rel_path)
    assert os.path.isabs(path) and path.endswith(rel_path)

def test_make_abs_abs():
    abs_path = os.path.abspath("test/hello.py")
    path = make_abs(abs_path)
    assert path == abs_path

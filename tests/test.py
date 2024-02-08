import pytest
from src.getters import my_functions as mf

def test_add():
    assert mf.add(1, 2) == 3
    
def test_divide():
    assert mf.divide(4, 2) == 2
    assert mf.divide(5, 2) == 2.5
    assert mf.divide(5, 0) == None
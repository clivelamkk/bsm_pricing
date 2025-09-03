import pytest
import numpy as np
from bsm_pricing import (
    validate_inputs,
    broadcast_inputs,
    gen_inverse_fut_local,
    gen_inverse_fut_usd,
    bsm_Nd2,
    general_bsm,
    inverse_bsm_local,
    general_bsm_iv
)

def test_validate_inputs():
    assert validate_inputs(1, [2, 3], np.array([4, 5])) == 2
    assert validate_inputs(1, None, 3.0) == 1
    with pytest.raises(ValueError):
        validate_inputs([1, 2], [1, 2, 3])

def test_broadcast_inputs():
    spots, vols = broadcast_inputs([100, 200], 0.2)
    assert len(spots) == 2 and len(vols) == 2
    assert np.array_equal(vols, np.array([0.2, 0.2]))
    spots, vols, is_call = broadcast_inputs(100, [0.2, 0.3], True)
    assert len(spots) == 2 and np.array_equal(spots, np.array([100, 100]))
    assert np.array_equal(is_call, np.array([True, True]))

def test_gen_inverse_fut_local():
    result = gen_inverse_fut_local(100, 100, greek_type=0)
    assert np.isclose(result, 0.0)
    result = gen_inverse_fut_local([100, 200], 100, greek_type=1)
    assert result.shape == (2, 3)
    assert np.allclose(result[:, 0], [0.0, -0.005])  # Price
    assert np.allclose(result[:, 1], [0.0001, 0.000025])  # Delta

def test_gen_inverse_fut_usd():
    result = gen_inverse_fut_usd(100, 100, greek_type=0)
    assert np.isclose(result, 0.0)
    result = gen_inverse_fut_usd([100, 200], 100, greek_type=1)
    assert result.shape == (2, 7)
    assert np.allclose(result[:, 0], [0.0, 1.0])  # Price
    assert np.allclose(result[:, 1], [0.01, 0.01])  # Delta

def test_bsm_Nd2():
    result = bsm_Nd2(100, 0.2, 100, 1.0, 0.05, 0.0, True)
    assert 0.0 <= result <= 1.0
    result = bsm_Nd2([100, 110], 0.2, 100, 1.0, 0.05, 0.0, [True, False])
    assert len(result) == 2
    assert np.all(result >= 0)

def test_general_bsm():
    result = general_bsm(100, 0.2, 100, 1.0, 0.05, True, 0.0, greek_type=0)
    assert isinstance(result, np.ndarray)
    result = general_bsm([100, 110], 0.2, 100, 1.0, 0.05, [True, False], 0.0, greek_type=1)
    assert result.shape == (2, 7)
    assert np.all(result[:, 0] >= 0)  # Price non-negative

def test_inverse_bsm_local():
    result = inverse_bsm_local(100, 0.2, 100, 1.0, 0.05, True, 0.0, greek_type=0)
    assert isinstance(result, np.ndarray)
    result = inverse_bsm_local([100, 110], 0.2, 100, 1.0, 0.05, [True, False], 0.0, greek_type=1)
    assert result.shape == (2, 7)
    assert np.all(result[:, 0] >= 0)  # Price non-negative

def test_general_bsm_iv():
    price = general_bsm(100, 0.2, 100, 1.0, 0.05, True, 0.0, greek_type=0)
    vol = general_bsm_iv(price, 100, 100, 1.0, 0.05, True, 0.0)
    assert np.isclose(vol, 0.2, atol=1e-4)
    vol = general_bsm_iv([price, price], [100, 110], 100, 1.0, 0.05, [True, False], 0.0)
    assert len(vol) == 2
    assert np.all(vol >= -1)
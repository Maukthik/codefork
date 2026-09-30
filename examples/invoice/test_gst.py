import pytest
from gst import gst_rate, split_gst


def test_known_rates():
    assert gst_rate("food") == 5
    assert gst_rate("luxury") == 28


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        gst_rate("spaceship")


def test_intrastate_split_is_half_each():
    assert split_gst(1000, 18) == {"cgst": 90.0, "sgst": 90.0}


def test_interstate_is_igst():
    assert split_gst(1000, 18, interstate=True) == {"igst": 180.0}

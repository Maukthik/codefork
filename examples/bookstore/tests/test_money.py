from bookstore.money import round_money


def test_half_up_rounding():
    assert round_money(2.675) == 2.68
    assert round_money(1.005) == 1.01


def test_whole_amounts_unchanged():
    assert round_money(10) == 10.0

import pytest

from bookstore.shipping import shipping_cost


@pytest.mark.parametrize("kg,cost", [(0.5, 40), (1.0, 40), (1.2, 65), (2.0, 65), (2.1, 90)])
def test_weight_tiers(kg, cost):
    assert shipping_cost(100, kg) == cost


def test_free_from_499():
    assert shipping_cost(499, 5) == 0

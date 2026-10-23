from datetime import date

import pytest

from library.loans import Library

DAY = date(2026, 10, 1)


def lib():
    l = Library()
    l.add_book("111", "Wings of Fire", copies=2)
    l.add_book("222", "Malgudi Days")
    return l


def test_checkout_sets_due_date_two_weeks_out():
    loan = lib().checkout("111", "asha", DAY)
    assert loan.due == date(2026, 10, 15)


def test_checkout_reduces_available_copies():
    l = lib()
    l.checkout("111", "asha", DAY)
    assert l.available("111") == 1


def test_cannot_borrow_when_no_copies_left():
    l = lib()
    l.checkout("222", "asha", DAY)
    with pytest.raises(ValueError):
        l.checkout("222", "ravi", DAY)


def test_return_puts_copy_back():
    l = lib()
    l.checkout("222", "asha", DAY)
    l.return_book("222", "asha", DAY)
    assert l.available("222") == 1


def test_adding_same_isbn_adds_copies():
    l = lib()
    l.add_book("222", "Malgudi Days", copies=3)
    assert l.available("222") == 4

"""Tests for the fake billing backend.

Pure data and pure functions, so no mocking. Two things get extra attention:
results must survive being sent to a model as JSON, and a bad lookup must come
back as data rather than an exception - the model is allowed to ask for
something that doesn't exist.
"""

import json

import pytest

from customer_service import tools


def test_get_invoice_returns_the_invoice():
    invoice = tools.get_invoice("inv_1042")
    assert invoice["amount"] == 29.00
    assert invoice["date"] == "2026-03-03"
    assert invoice["status"] == "paid"


def test_an_unknown_invoice_is_an_error_not_an_exception():
    """The model can guess an id. That must not kill the turn."""
    assert "error" in tools.get_invoice("inv_does_not_exist")


def test_list_invoices_returns_the_whole_history_newest_first():
    invoices = tools.list_invoices(tools.DEMO_CUSTOMER_ID)["invoices"]
    dates = [invoice["date"] for invoice in invoices]
    assert dates == sorted(dates, reverse=True)
    assert len(invoices) == 4


def test_an_unknown_customer_is_an_error_not_an_exception():
    assert "error" in tools.list_invoices("cus_nobody")
    assert "error" in tools.get_subscription("cus_nobody")


def test_get_subscription_returns_the_current_plan():
    subscription = tools.get_subscription(tools.DEMO_CUSTOMER_ID)
    assert subscription["plan"] == "Pro"
    assert subscription["status"] == "active"
    assert subscription["renews_on"] == "2026-04-03"


def test_the_duplicate_charge_is_really_there():
    """The seeded scenario: 'I was charged twice' must be findable and true.

    If this fails, the demo's headline case silently becomes a hallucination
    test instead of a lookup test.
    """
    invoices = tools.list_invoices(tools.DEMO_CUSTOMER_ID)["invoices"]
    march = [invoice for invoice in invoices if invoice["date"] == "2026-03-03"]

    assert len(march) == 2
    assert {invoice["amount"] for invoice in march} == {29.00}


@pytest.mark.parametrize(
    "result",
    [
        tools.get_invoice("inv_1042"),
        tools.get_invoice("nope"),
        tools.list_invoices(tools.DEMO_CUSTOMER_ID),
        tools.get_subscription(tools.DEMO_CUSTOMER_ID),
    ],
)
def test_every_result_survives_the_trip_to_the_model(result):
    """Tool results are sent back as JSON, so they have to serialise."""
    assert json.loads(json.dumps(result)) == result


def test_callers_cannot_corrupt_the_backing_data():
    """Results are copies. A caller mutating one must not edit the 'database'."""
    invoice = tools.get_invoice("inv_1042")
    invoice["amount"] = 999.99

    assert tools.get_invoice("inv_1042")["amount"] == 29.00


def test_billing_tools_are_callable_and_documented():
    """Docstrings become the description the model reads when choosing a tool."""
    assert tools.BILLING_TOOLS
    for tool in tools.BILLING_TOOLS:
        assert callable(tool)
        assert tool.__doc__, f"{tool.__name__} needs a docstring - the model reads it"

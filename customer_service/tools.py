"""In-memory stand-in for the billing system the agents look things up in.

Plain data and plain functions: no model, no network, no database. A real
deployment would put an HTTP client behind these signatures. The point for now
is that agents answer from actual records instead of inventing them.

Each function's docstring is not decoration - it becomes the description the
model reads when deciding which one to call, so it is written for that reader.
"""

import copy
from collections.abc import Callable

# The one customer this demo knows about. Identifying the customer from the
# conversation would be its own feature; the CLI hands this to the agent instead.
DEMO_CUSTOMER_ID = "cus_demo"

_CUSTOMERS: dict[str, dict] = {
    "cus_demo": {
        "customer_id": "cus_demo",
        "name": "Alex Rivera",
        "email": "alex@example.com",
        "since": "2025-11-14",
    },
}

# Seeded so that "I was charged twice" is a real, findable problem: inv_1042 and
# inv_1043 are the same amount on the same day. An agent that looks will find it;
# one that guesses will not.
_INVOICES: dict[str, dict] = {
    "inv_1040": {
        "invoice_id": "inv_1040", "customer_id": "cus_demo", "amount": 29.00,
        "currency": "USD", "date": "2026-01-03", "status": "paid",
        "description": "Pro plan - January",
    },
    "inv_1041": {
        "invoice_id": "inv_1041", "customer_id": "cus_demo", "amount": 29.00,
        "currency": "USD", "date": "2026-02-03", "status": "paid",
        "description": "Pro plan - February",
    },
    "inv_1042": {
        "invoice_id": "inv_1042", "customer_id": "cus_demo", "amount": 29.00,
        "currency": "USD", "date": "2026-03-03", "status": "paid",
        "description": "Pro plan - March",
    },
    "inv_1043": {
        "invoice_id": "inv_1043", "customer_id": "cus_demo", "amount": 29.00,
        "currency": "USD", "date": "2026-03-03", "status": "paid",
        "description": "Pro plan - March",
    },
}

_SUBSCRIPTIONS: dict[str, dict] = {
    "cus_demo": {
        "customer_id": "cus_demo",
        "plan": "Pro",
        "price": 29.00,
        "currency": "USD",
        "interval": "monthly",
        "status": "active",
        "renews_on": "2026-04-03",
    },
}


def get_invoice(invoice_id: str) -> dict:
    """Look up one invoice by its id, for example inv_1042.

    Returns the invoice with its amount, date, status and description, or an
    error if no invoice has that id.
    """
    invoice = _INVOICES.get(invoice_id)
    if invoice is None:
        # An error is data, not an exception: this goes back to the model, which
        # can then ask the customer for a correct id instead of the turn dying.
        return {"error": f"No invoice with id {invoice_id}."}
    return copy.deepcopy(invoice)


def list_invoices(customer_id: str) -> dict:
    """List every invoice for a customer, newest first.

    Use this to check a customer's charge history - for example to see whether
    they were billed twice for the same thing.
    """
    if customer_id not in _CUSTOMERS:
        return {"error": f"No customer with id {customer_id}."}

    invoices = [invoice for invoice in _INVOICES.values() if invoice["customer_id"] == customer_id]
    invoices.sort(key=lambda invoice: invoice["date"], reverse=True)
    return {"invoices": copy.deepcopy(invoices)}


def get_subscription(customer_id: str) -> dict:
    """Get a customer's current plan, price, status and next renewal date."""
    subscription = _SUBSCRIPTIONS.get(customer_id)
    if subscription is None:
        return {"error": f"No subscription for customer {customer_id}."}
    return copy.deepcopy(subscription)


# What the billing agent is allowed to call. The technical agent gets none of
# these - it has nothing to look up yet.
BILLING_TOOLS: tuple[Callable[..., dict], ...] = (get_invoice, list_invoices, get_subscription)

"""Detector label -> our plain item names (mcp_server.store.norm). Labels taken from classifier/image_model/vocabulary.py."""
import pytest
from mcp_server.store import norm


@pytest.mark.parametrize("raw,expected", [
    ("tomato", "tomato"), ("Tomatoes", "tomato"), ("cherry tomato", "tomato"),
    ("onion", "onion"), ("red onion", "onion"), ("spring onion", "onion"), ("shallot", "onion"),
    ("green capsicum", "capsicum"), ("red capsicum", "capsicum"), ("bell pepper", "capsicum"),
    ("egg", "eggs"), ("eggs", "eggs"), ("boiled egg", "eggs"),
    ("curd", "yogurt"), ("greek yogurt", "yogurt"), ("yogurt", "yogurt"),
    ("milk", "milk"), ("whole milk", "milk"), ("almond milk", "milk"), ("milk bottle", "milk"),
    ("cheddar cheese", "cheese"), ("mozzarella cheese", "cheese"), ("cheese", "cheese"), ("paneer", "paneer"),
    ("white bread", "bread"), ("brown bread", "bread"), ("sourdough bread", "bread"),
    ("chicken breast", "chicken"), ("chicken thighs", "chicken"), ("raw chicken", "chicken"),
    ("baby spinach", "spinach"), ("green apple", "apple"), ("bananas", "banana"),
    ("tomato_soup", "tomato soup"),                 # a dish, not a tomato
    ("peanut butter", "peanut butter"),             # not butter
    ("sweet potato", "sweet potato"),
    ("parsley", "parsley"), ("", ""),
])
def test_norm(raw, expected):
    assert norm(raw) == expected

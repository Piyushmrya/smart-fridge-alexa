"""Fake classifier: pushes a random fridge snapshot to the MCP server every few seconds via the real ingest API.
The real classifier replaces this -- see classifier_stub/push_example.py."""
import os, random, sys, time
from classifier_stub.push_example import push

POOL = ["milk", "eggs", "spinach", "tomato", "onion", "paneer", "cheese", "bread", "yogurt",
        "carrot", "apple", "banana", "capsicum", "chicken", "potato", "cucumber", "butter"]

if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    while True:
        items = [{"item": i, "qty": random.randint(1, 4), "confidence": round(random.uniform(0.7, 0.98), 2)}
                 for i in random.sample(POOL, random.randint(5, 10))]
        print(push(items, source="mock"))
        time.sleep(interval)

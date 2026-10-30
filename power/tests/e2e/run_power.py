"""vllm-power for the end-to-end test: the REAL service (fleet_power.__main__) with the suite's inventory
(tests/fakes.py, POWER_BOXES) and only the box ADDRESSES pointed at containers on the isolated test
network (E2E_HOSTS="box-a=ip,box-b=ip,..."). Keys, known_hosts, gate secret, origins, audit and dry-run
come from the same env as production."""
import json
import os

import fleet_power.__main__ as m
from tests.fakes import TEST_INVENTORY

hosts = dict(kv.split("=", 1) for kv in os.environ["E2E_HOSTS"].split(","))
inventory = json.loads(TEST_INVENTORY)
for box in inventory:
    if box["name"] in hosts:
        box["ssh_host"] = hosts[box["name"]]
os.environ["POWER_BOXES"] = json.dumps(inventory)
m.main()

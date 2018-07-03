import json
import sys

nb = sys.stdin.read()
nb = json.loads(nb)
md = nb["metadata"]
clean_output = False
if "mlworkflow" in md:
    if "clean_output" in md["mlworkflow"] and md["mlworkflow"]["clean_output"]

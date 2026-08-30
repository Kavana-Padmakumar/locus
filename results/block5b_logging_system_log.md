\# Block 5B — Config + Logging System Log



Date: 2026-08-30

Config source: configs/corruption\_matrix.yaml (loaded via src/config\_loader.py)

Logger: src/results\_logger.py, writes structured CSV + JSONL keyed by

&#x20;       (method, corruption, severity, seed)



Verification: Re-ran the no-adaptation baseline through the new config-driven

pipeline (src/eval\_no\_adaptation\_v2.py). Results in baseline\_results\_v2.csv

match Sunday's baseline\_results.csv within floating-point tolerance.

Note: seed is loaded from config but passed as None in eval\_no\_adaptation\_v2.py — 

no-adaptation eval is deterministic (model.eval(), no\_grad, shuffle=False), so this 

has no effect on results. Seed will matter starting with stochastic methods (TENT).



Note: cfg\["methods"] (BN-Adapt, TENT) is loaded by config\_loader.py but not yet 

consumed — this script only runs the no\_adaptation baseline. Method-driven dispatch 

is pending for the block that adds adaptive methods.



Status: COMPLETE.


\# Block 8C — Dual-Parameter-State Engine Hardening Log



Date: 2026-09-10



Task: Buffer/hardening day for Block 8 (the project's own risk table

flags this as the highest-risk piece). Block 8B already passed, so this

day adds defense-in-depth tests anyway, per the block's own instruction.

Implemented in src/test\_dual\_state\_hardening.py.



Pilot condition: defocus\_blur, severity 3 - a third condition, distinct

from Block 8A/8B's gaussian\_noise/contrast, for broader coverage.



Results:

\[Test 1] Repeated identical-subset calls determinism: True (theta left unchanged: True) -> PASS

\[Test 2] Duplicate layer names harmless: True -> PASS

\[Test 3] Unknown layer name raises correctly: True -> PASS

\[Test 4] Partial patch differs from both extremes: True (differs from zero-patch: True, differs from full-patch: True) -> PASS



Tests passed: 4/4



Status: COMPLETE.


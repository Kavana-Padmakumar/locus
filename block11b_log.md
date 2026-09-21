\# Block 11B — Cache CKA pipeline outputs to disk

Date: 2026-09-16



\## Task

Cache the CKA pipeline's outputs to disk (avoid recompute across later runs).

Run the identity sanity check: CKA between a model and itself must equal 1.0 at every layer.



\## Cold run (cache deleted, real computation)

20/20 layers pass identity check.

CKA(model, itself) = 1.0000000000 at every layer.

Output saved to: block11b\_cold\_output.txt



\## Warm run (cache present, should read from disk)

20/20 layers pass identity check.

CKA(model, itself) = 1.0000000000 at every layer.

Output saved to: block11b\_warm\_output.txt



\## Cache correctness check

Compare-Object between block11b\_cold\_output.txt and block11b\_warm\_output.txt

Result: zero differences (empty output) — cold and warm runs are byte-identical.



\## Cache files created

<COUNT> .json files in cache/cka



\## Definition of Done

\- CKA(model, itself) = 1.0 at every layer: PASSED

\- Probe set frozen and version-controlled: CONFIRMED (verified Sep 15, data/cka\_probe\_set.pt live on GitHub)



\## Status: COMPLETE


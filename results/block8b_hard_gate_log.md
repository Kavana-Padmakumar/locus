\# Block 8B — Dual-Parameter-State Hard-Gate Correctness Test Log



Date: 2026-09-09



Task: Write and run the mechanical correctness check - patching ALL BN

layers from theta to theta-prime must exactly reproduce theta-prime's

own output. This is a hard gate for the entire causal-patching

methodology used in later phases. Implemented in

src/test\_dual\_state\_hard\_gate.py.



Tested on two independent pilot conditions (gaussian\_noise sev5,

contrast sev5) so a pass isn't a fluke of one specific input/adaptation.

Bug found and fixed before this result: DualStateModel and the test

script both called .eval() on the adapted model, which made BatchNorm

use running statistics instead of per-batch statistics - inconsistent

with how BN-Adapt/TENT actually operate (see eval\_bn\_adapt.py,

eval\_tent.py). Switched both to .train() with requires\_grad\_(False),

matching the rest of the project's convention. This produced the exact

match below; the pre-fix run showed max differences of \~2.4-2.9 on raw

logits, which correctly failed the gate rather than passing on a broken

comparison.



Results:

gaussian\_noise, sev5: matches=True  max\_diff=0.00e+00

contrast, sev5:       matches=True  max\_diff=0.00e+00



Hard gate: PASSED



Status: COMPLETE.


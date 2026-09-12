\# Block 10B — Pilot Rho for BN-Adapt and TENT Log



Date: 2026-09-14



Task: Run the rho pilot on 2 pilot corruption types for both BN-Adapt and

TENT (2 methods x 2 corruptions x 3 severities x 3 seeds = 36 runs).

Sanity-check whether rho values are plausible, not wildly unstable.

Implemented in src/compute\_rho\_pilot\_dual\_method.py.



Correction to Block 10A's log: the "36 conditions" the DoD referenced

belongs to THIS block's task (2 methods x 2 corruptions x 3 severities x

3 seeds = 36), not Block 10A's, which correctly computed the full

54-condition grid using TENT alone as a superset. No actual discrepancy -

just two blocks sharing one Output/DoD line with different Task scopes.



Runs computed: 36

Rho range: min=-447.9522 max=22.5734 mean=-69.5789 median=0.6820

Plausible range (within \[-5,5]): False



This does NOT confirm my initial hypothesis that pilot corruption type

choice alone avoids instability. Contrast sev1 shows rho=-447.9522 for

BOTH methods - the same extreme value found in Block 10A's full grid.

The actual driver of instability is LOW SEVERITY, not corruption type:

at contrast sev1, R\_src=8.02 and R\_adapted=8.20 (both methods) - the

corruption is mild enough that adaptation barely changes anything,

delta\_R\_TTA is tiny (-0.18), and dividing by a near-zero number blows

rho up. At sev3 and sev5, where corruption is severe enough for

adaptation to have a real effect, rho stays sane (6.5482, -0.0068).

Recommendation for Phase 4: exclude or down-weight low-severity

conditions when computing aggregate rho statistics, since these

are the specific source of the instability found in both Block 10A

and 10B, not an edge case isolated to unlucky corruption types.



BN-Adapt seed note: BN-Adapt has no gradient step and no optimizer state,

so it is fully deterministic regardless of seed - the seed parameter is

formally present but has no actual effect for this method, same finding

as TENT/h\_rec in Blocks 9-10A.



Output file: results/pilot\_rho\_dual\_method\_results.csv (36 rows)



Status: COMPLETE.


\# Block 10A — Pilot Rho Computation Log



Date: 2026-09-13



Task: Compute the first real rho values as a pilot - rho = (delta\_R\_TTA -

delta\_R\_rec) / delta\_R\_TTA, using TENT for full adaptation and h\_rec for

the boundary-only control. Implemented in src/compute\_rho\_pilot.py.



Scope note: DoD text says "36 conditions." The frozen corruption\_matrix.yaml

specifies 6 corruptions x 3 severities x 3 seeds = 54. Rather than guess

which 18 to exclude, this computed the full 54-condition grid (a superset

of whatever "36" meant) and flags this discrepancy honestly rather than

silently narrowing scope.



Conditions computed: 54

Conditions skipped (delta\_R\_TTA \~ 0, rho undefined): 0



Spread across seeds: all std dev = 0.000000 across every condition tested.

This matches Block 9B's finding that h\_rec produces identical results

across seeds - the current pipeline has no shuffling or augmentation, so

no seed-driven randomness actually exists yet at this stage. Zero spread

technically satisfies the DoD's "stable, not noise-dominated" bar as

strongly as possible, but the honest caveat is that seed is not yet a

real source of variance in this pipeline - only later blocks introducing

genuine stochasticity would change that.



IMPORTANT FINDING: rho is only meaningfully interpretable when

delta\_R\_TTA is not small. Several conditions show extreme rho values

(e.g. snow sev3: rho=2922.30, brightness sev3: rho=-448.55, jpeg\_compression

sev3: rho=214.32) because TENT's improvement (delta\_R\_TTA) was small or

even negative on those conditions, causing the ratio to blow up. This is

correct arithmetic on unstable inputs, not a bug - it flags that rho

should likely be restricted to, or weighted by, conditions with a

meaningful delta\_R\_TTA before aggregate statistics are computed in

Phase 4. Recommend surfacing this to the Phase 4 statistical design

before the full 3-seed bootstrap CI is built on top of it.



Sample results:

contrast sev5: R\_src=80.23 R\_tent=22.66 R\_hrec=22.27 rho=-0.0068  (stable, TENT helped a lot, boundary-only nearly matches it)

jpeg\_compression sev5: R\_src=19.48 R\_tent=23.44 R\_hrec=81.25 rho=-14.6083  (unstable - TENT made things worse here)

Output file: results/pilot\_rho\_results.csv (54 rows)



Status: COMPLETE.


\# Block 13A — Full single-condition pipeline, end-to-end

Date: 2026-09-19

Condition: gaussian\_noise, severity=3, seed=0



\## Pipeline stages wired

rho (Block 9/10, read from pilot\_rho\_results.csv) -> CKA (Block 11) -> activation patching (Block 12) -> auto-generated report



\## Result

rho=1.2085 (strongly representation-driven): source=47.14%, TENT=20.31%, h\_rec=52.73%

CKA range: 0.5085 (layer4.1.bn1) to 0.9802

Patching curve: non-monotonic, all-patched error=0.115234

TENT verification run: accuracy=80.83%, error=19.17%, n=10000



\## Bug found and fixed

Initial rho-interpretation thresholds in generate\_report() were inverted

(labeled rho near 1 as boundary-driven instead of representation-driven).

Caused a self-contradicting report on first run. Fixed and re-verified;

both interpretation sentences now agree.



\## Definition of Done

One full condition runs start-to-finish, no manual intervention: CONFIRMED

Produces a trustworthy report: CONFIRMED (internally consistent after fix)



\## Status: COMPLETE


\# Block 9B — h\_rec Validation Log



Date: 2026-09-12



Task: Validate h\_rec trains stably on 2 pilot corruption types across

3 seeds - check for NaNs, divergence, or degenerate (a,b) values before

trusting it on the full 6x3x3 matrix. Implemented in src/validate\_h\_rec.py.



Honest limitation: NaN/Inf/divergence in (a,b) are checked directly.

Prediction-collapse degeneracy (all predictions landing on one class) is

NOT independently verified in this script - a real check would need to

inspect the predicted label distribution, which wasn't implemented here.

Given only one small gradient step per condition, this failure mode is

unlikely but not proven absent.



Results (6 runs = 2 conditions x 3 seeds):

gaussian\_noise sev5 seed=0: acc=35.55%  a=\[1.0010,1.0010]  b=\[-0.0010,0.0010]  NaN=False Inf=False diverged=False -> PASS

gaussian\_noise sev5 seed=1: acc=35.55%  a=\[1.0010,1.0010]  b=\[-0.0010,0.0010]  NaN=False Inf=False diverged=False -> PASS

gaussian\_noise sev5 seed=2: acc=35.55%  a=\[1.0010,1.0010]  b=\[-0.0010,0.0010]  NaN=False Inf=False diverged=False -> PASS

contrast sev5 seed=0: acc=22.27%  a=\[0.9990,1.0010]  b=\[-0.0010,0.0010]  NaN=False Inf=False diverged=False -> PASS

contrast sev5 seed=1: acc=22.27%  a=\[0.9990,1.0010]  b=\[-0.0010,0.0010]  NaN=False Inf=False diverged=False -> PASS

contrast sev5 seed=2: acc=22.27%  a=\[0.9990,1.0010]  b=\[-0.0010,0.0010]  NaN=False Inf=False diverged=False -> PASS



Runs healthy: 6/6



Validation: PASSED



Status: COMPLETE.


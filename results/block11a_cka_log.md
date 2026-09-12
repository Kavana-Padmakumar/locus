\# Block 11A — CKA Pipeline Implementation Log



Date: 2026-09-15



Task: Implement linear CKA across all normalization layers using a

fixed, deterministic probe set, frozen once for comparability across

every later block. Implemented in src/cka.py, probe set frozen via

src/freeze\_probe\_set.py, identity check in src/test\_cka\_identity.py.



Probe set: 512 images, seed=42, saved to

data/cka\_probe\_set.pt, committed to git - must never be regenerated

with a different seed, or all future CKA comparisons become invalid.



Identity check (CKA(model, itself) = 1.0 at every layer):

Layers checked: 20

Layers passed: 20

Sample values: bn1: 1.0000000000, layer1.0.bn1: 1.0000000000,

layer2.0.downsample.1: 1.0000000000, layer4.1.bn2: 1.0000000000



Note: initial implementation crashed with a 32GB memory allocation error

(feature-covariance formulation blows up for early layers with large

spatial x channel dimensions). Fixed by reformulating linear\_cka to use

the Gram-matrix (images x images) formulation instead, which stays small

(512x512) regardless of layer size. Fix verified: identity check now

passes cleanly with no memory issues.



Status: COMPLETE.


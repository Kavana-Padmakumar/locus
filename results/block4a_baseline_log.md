\# Block 4A — Baseline Verification Log



Date: 2026-08-27 (session extended into later attempts due to network issues)

Checkpoint: huyvnphan/PyTorch\_CIFAR10, resnet18

Published clean accuracy: 93.07%



Status: PARTIAL — blocked on network conditions, not a code or setup problem.



Completed and verified:

\- Compute environment set up (PyTorch, torchvision installed locally)

\- Repository cloned, weights downloaded and extracted correctly (resnet18.pt confirmed

&#x20; present in cifar10\_models/state\_dicts/)

\- Model loads successfully with pretrained=True, no errors

\- baseline\_eval.py script written and confirmed syntactically correct



Blocked:

\- CIFAR-10 test-set download failed/stalled across 6 different sources and methods

&#x20; (torchvision's own downloader via Colab and locally, Google's Keras-hosted mirror,

&#x20; the official Toronto mirror via both Invoke-WebRequest and curl) - all showing severe

&#x20; slowdowns or truncation. This points to a local network condition today, not a

&#x20; server-side issue, given the consistent pattern across unrelated hosts.



Next step: re-run python baseline\_eval.py once network conditions normalize. No code

changes needed - the pipeline is fully built and just needs the data to actually download.


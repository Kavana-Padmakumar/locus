# LOCUS Literature Notes
Started: 2026-08-18

## 1. TENT (Wang et al., ICLR 2021) — arXiv:2006.10726

**Core mechanic:**
TENT adapts a pretrained model at test time using only unlabeled batches. It minimizes the
mean prediction entropy over each test batch, updating only the BatchNorm affine parameters
(gamma, beta) plus re-estimating BN running statistics on the batch. No labels, no source
data, no backprop through the rest of the network — one gradient step per batch, computed
online, without altering the original training.

**THE key passage — calibration ambiguity:**
Location: Page 9, Section 6 ("Discussion"), subheading "Losses" — last sentence of that
paragraph, right before the closing line about tent "yielding a new model with every
update." (Verified directly against the PDF, ICLR 2021 camera-ready, arXiv:2006.10726.)

Exact sentence (verified from full text):
"Returning to entropy specifically, this loss may interact with calibration (Guo et al.,
2017), as better uncertainty estimation could drive better adaptation."

This is a direct citation to Guo, Pleiss, Sun, Weinberger — "On Calibration of Modern
Neural Networks," ICML 2017 (the paper that introduced temperature scaling / ECE as the
standard calibration framework).

**Why this matters for LOCUS:**
This is the anchor citation for the entire project's research gap. The TENT authors
themselves flag — as a forward-looking, unresolved remark, not a tested claim — that their
entropy-minimization gains might be explainable by improved calibration rather than genuine
representation learning. They never test this. Nobody has, as far as this search found.
LOCUS is, functionally, the paper that goes back and actually checks the sentence above.

---

## 2. BN-Adapt (Schneider et al., NeurIPS 2020) — arXiv:2006.16971

**Core mechanic:**
Re-estimates BatchNorm running mean/variance using unlabeled target-domain batches, with
zero gradient steps — purely a statistics swap at inference time. No entropy loss, no
optimization at all; it's the simplest possible member of the TTA family.

**Relevant finding:**
Concrete headline number: replacing training-set BN statistics with corrupted-batch
statistics took ResNet-50 from 76.7% mCE to 62.2% mCE on ImageNet-C — a large jump from a
one-line change. The paper frames this mechanistically as evidence that "the features upon
which the models rely are still present in the corrupted images" — i.e., they interpret the
gain as restoring access to features that were already there, not as building new ones. This
representation-preserving framing is itself informative for LOCUS: it's a testable claim,
not a demonstrated one.

---

## 3. BN-Adapt companion (Nado et al., 2020) — arXiv:2006.10963
"Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift"

**Why I'm reading this one too:**
This is the strongest existing bridge between BN-based TTA and calibration. It directly
measures both accuracy AND calibration (ECE, Brier score) under prediction-time BN, achieving
60.28% mCE on ImageNet-C — competitive with Schneider et al.'s concurrent result.

**Key finding to extract:**
Related-work section states plainly: "Guo et al. (2017) observed that models using
traditional batch norm typically have worse calibration on the test set, [but] to our
knowledge no one has applied normalization strategies for correcting miscalibration under
covariate shift." This confirms the calibration angle was recognized as a live, open thread
as early as 2020 — three papers (TENT, Schneider, Nado) all circling the same
representation-vs-calibration ambiguity, and none of them resolving it with a causal test.

---

## 4. SAR (Niu et al., ICLR 2023 Oral, top-5%) — arXiv:2302.12400

**Core mechanic:**
Sharpness-Aware and Reliable entropy minimization. Built to fix a failure mode TENT has:
under small batch sizes / noisy real-world streams, a few samples with very large gradients
can drive the model into a collapsed trivial solution (predicting one class for everything).
SAR fixes this two ways: (1) drops samples whose gradient norm is too large before the
entropy step, (2) adds sharpness-aware optimization so the model settles into a flatter,
more robust minimum.

**Why this matters for LOCUS:**
SAR's own motivation is explicitly about a failure mode, not the calibration question — but
it's useful background for H2 (small-batch collapse, STRETCH-tier in the current plan). Their
framing of collapse as "large-gradient noisy samples destabilizing adaptation" is a
parameter-space explanation; LOCUS's H2 asks a complementary question — WHERE in the network
(boundary region vs. representation) that destabilization actually shows up.

---

## 5. Gap statement

Nobody has actually tested the thing three separate papers all half-noticed. TENT's authors
wonder out loud, in one sentence, if their method's gain might really be about calibration
instead of the model getting smarter. Nado et al. say flat out that no one has used
normalization to fix miscalibration under shift. Schneider et al. just assume their BN fix
is "giving the model its features back" without checking if that's actually what's
happening. Three groups, same suspicion, zero experiments. LOCUS is the experiment: split
the accuracy gain into a part a boundary-only model can reproduce on its own and a part it
can't, then use patching and CKA to prove — not guess — which layers the un-reproducible
part is actually coming from.

---

## 6. Adjacent/close-but-not-identical papers spotted while searching

- "Towards Test Time Adaptation via Calibrated Entropy Minimization" (KDD 2024) — title
  alone confirms someone else has also noticed the calibration angle is worth engaging
  directly, from a different (method-design, not causal-diagnostic) direction. Worth reading
  the abstract in Block 1B tomorrow to confirm it doesn't already do what LOCUS proposes.
- "On Pitfalls of Test-Time Adaptation" (Zhao et al., ICML 2023, arXiv:2306.03536) —
  queued for tomorrow (Block 2), not read yet today.

---

## 7. TTA Survey (Liang, He, Tan — arXiv:2303.15361, IJCV 2024)

**Taxonomy confirmed:**
The survey splits all TTA methods into three categories — TTDA (whole test set), TTBA
(mini-batch), OTTA (streaming) — purely by data-access pattern. No category, anywhere in
the taxonomy, is organized around causal mechanism (representation vs. boundary). This is
the clearest confirmation yet that the entire field's organizing framework has never asked
LOCUS's question.

**Why this matters for LOCUS:**
A comprehensive survey covering the whole field, as of 2023-2024, had every opportunity to
notice the representation-vs-boundary distinction as a category worth taxonomizing by. It
didn't. That's not proof no one has thought about it — but it's strong evidence the question
sits outside how the field currently organizes its own knowledge.

---

## 8. On Pitfalls of Test-Time Adaptation (Zhao et al., ICML 2023) — arXiv:2306.03536

**Three pitfalls identified (via their TTAB benchmark):**
1. Hyperparameter/model-selection is difficult due to online batch dependency
2. TTA effectiveness varies significantly with pretrained model quality
3. No existing method handles all distribution shift types well

**Why this matters for LOCUS:**
All three pitfalls are about *when* TTA works, not *why* it works when it does. Even a
paper explicitly designed to stress-test TTA methods and expose their weaknesses never
asks the representation-vs-boundary question — it only asks "does accuracy hold up."
Reinforces that this is a genuine blind spot, not just an oversight in one or two papers.

---

## 9. Updated gap statement (revised after reading survey + pitfalls papers)

The gap isn't just that three early papers (TENT, Schneider, Nado) noticed and dropped the
calibration question. It's that the field's own comprehensive taxonomy (Liang et al.) and
its own most rigorous stress-test (Zhao et al.) both organize TTA entirely around
data-access patterns and failure conditions — never around causal mechanism. Five papers
spanning 2020-2024, including the field's survey and its most critical benchmark paper,
all miss the same question. LOCUS is the first attempt to ask it directly and answer it
causally rather than by inference from accuracy numbers alone.


---

## 10. Mechanistic interpretability literature — activation patching & CKA (Block 2A)

Six papers reviewed: ROME (Meng et al. 2022), causal mediation analysis (Vig et al. 2020),
causal abstraction (Geiger et al. 2021), activation patching best-practices (Heimersheim &
Nanda 2024; Zhang & Nanda 2024), and CKA (Kornblith et al. 2019), plus its vision
application (Raghu et al. 2021, ViT vs CNN comparison).

**Confirmed pattern across all six:** every one applies its causal/similarity tool to a
model that is static during the analysis — either a single frozen forward pass, or a
comparison between two separately-trained-but-fixed models. None intervene on a model
whose weights are changing online, mid-inference, in response to the very data being
analyzed. This is exactly the gap LOCUS needs: TTA models are the first case where the
thing being causally probed is not holding still.

**Engineering implication flagged for Phase 3:** since no prior activation-patching
implementation has been built for a two-weight-state system (source theta vs. adapted
theta-prime), the dual-parameter-state forward pass (Block 8) has no existing reference
implementation to build from — this is genuinely new engineering, not an adaptation of
an off-the-shelf tool.
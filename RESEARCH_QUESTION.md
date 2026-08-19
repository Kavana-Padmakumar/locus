# LOCUS — Final Research Question

## Research Question

For BN-Adapt and TENT applied to a ResNet-18 on CIFAR-10-C, is the majority of the
target-domain error reduction caused by a genuine change in the model's feature
representation, or is it reproducible by a capacity-matched decision-boundary-only
control that is architecturally prevented from touching the representation at all?

## Narrowed H1

The majority of BN-Adapt/TENT's accuracy gain on CIFAR-10-C is causally mediated by
boundary-region changes near the classifier, and is reproducible by a boundary-only
counterfactual control. Representational change (measured via CKA) will be largest in
early-to-middle layers, but activation patching will show this change to be largely
causally inert with respect to the accuracy improvement. This hypothesis is falsifiable
in both directions: if the boundary-only control cannot reproduce the gain, that is
equally strong evidence of genuine representation repair, and equally publishable.

## Why this is answerable within MUST-scope alone

This question requires only: (1) BN-Adapt and TENT implementations, (2) the boundary-only
control, (3) CKA per layer, (4) activation patching per layer, (5) the statistical
correlation test between the two — all scoped inside Phase 3 (Core Methodology) and
Phase 4 (Main Experiments) of the MUST tier. It has no dependency on SAR, ImageNet-C,
PACS/OfficeHome, ViT generalization, or the H2 collapse experiment — all of which remain
SHOULD/STRETCH and can be cut entirely without invalidating this question or its answer.

## Grounding

This question sits in a gap confirmed across nine papers read in Blocks 1 and 2:
- TENT (Wang et al. 2021), BN-Adapt (Schneider et al. 2020), and its companion
  (Nado et al. 2020) all independently flag the calibration/representation ambiguity
  without testing it.
- The field survey (Liang et al. 2023) and the most rigorous stress-test (Zhao et al. 2023)
  both organize TTA entirely by data-access pattern and failure mode, never by causal
  mechanism.
- Six foundational interpretability papers (Meng et al. 2022; Vig et al. 2020; Geiger et al.
  2021; Heimersheim & Nanda 2024; Zhang & Nanda 2024; Kornblith et al. 2019) all apply
  causal/similarity tools exclusively to static models — none to a self-modifying system.

LOCUS sits at the unclaimed intersection of these two literatures.

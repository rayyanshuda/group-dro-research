# Decomposing the Group DRO Advantage: Balanced Sampling vs Adversarial Reweighting on Waterbirds

This repo reproduces the Group DRO algorithm from:

Shiori Sagawa\*, Pang Wei Koh\*, Tatsunori Hashimoto, and Percy Liang.
[Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731). ICLR, 2020.

on the **Waterbirds** benchmark specifically (this repo does not reproduce the original paper's CelebA or MultiNLI experiments), and extends it with an original ablation: isolating how much of Group DRO's worst-group accuracy improvement comes from its group-balanced training sampler alone, before the adversarial reweighting is added on top.

The full writeup is in [`paper/paper.pdf`](paper/paper.pdf).

## Abstract

Neural networks trained with standard empirical risk minimization (ERM) can achieve strong average accuracy while failing on subpopulations where a spurious correlation in the training data does not hold. Group Distributionally Robust Optimization (Group DRO) addresses this failure mode by combining two different mechanisms: a group-balanced training sampler and an adversarial, worst-group-weighted loss. Published evaluations generally report only their combined effect, so the individual contribution of each mechanism remains unclear. I reproduced Group DRO on the Waterbirds benchmark, training a ResNet-50 under the paper’s reference configuration, and isolated each mechanism’s contribution by additionally training a third configuration that uses the balanced sampler alone, with no adversarial reweighting. My ERM and Group DRO baselines obtain 97.25% average / 68.54% worst-group test accuracy and 95.40% average / 86.14% worst-group accuracy, respectively; a +17.60-point worst-group improvement at a cost of 1.85 points of average accuracy, which matches the shape, though not the exact magnitude of the original paper’s result. The balanced-sampling-only ablation reaches 83.02% worst-group accuracy, recovering 82% of Group DRO’s total worst-group gain over ERM; adding adversarial reweighting produces the remaining 18% of the improvement in worst-group accuracy, with the additional gain concentrated on the rarest training group. This result comes from a single training run per configuration, due to the lack of compute available for this project. It is not a claim of statistical certainty. On this benchmark, balanced sampling alone recovers most of Group DRO’s practical benefit at a fraction of its implementation complexity, which describes a larger risk in evaluating combined interventions: a method’s reported aggregate effect can obscure how much of that effect comes from its simplest component.

## Results

**ERM baseline** (test accuracy):

| Model selected by | Avg. accuracy | Worst-group accuracy |
|---|---|---|
| Best val worst-group acc | 97.25% | **68.54%** |
| Best val average acc | 97.33% | 60.12% |
| Sagawa et al. (2020), Table 1 | 97.3% | 72.6% |

**Group DRO** (test accuracy):

| Model selected by | Avg. accuracy | Worst-group accuracy |
|---|---|---|
| Best val worst-group acc | 95.40% | **86.14%** |
| Best val average acc | 97.57% | 70.72% |
| Sagawa et al. (2020), Table 1 (standard reg.) | 97.4% | 76.9% |

**Decomposing the gain** (best-val-worst-group checkpoint, all arms):

| Arm | Avg. accuracy | Worst-group accuracy | Δ avg vs. ERM | Δ worst-group vs. ERM |
|---|---|---|---|---|
| ERM (no reweighting, plain cross-entropy) | 97.25% | 68.54% | — | — |
| Balanced-sampling-only | 96.28% | **83.02%** | −0.97 pts | **+14.48 pts** |
| Full Group DRO | 95.40% | **86.14%** | −1.85 pts | **+17.60 pts** |

Balanced sampling by itself recovers **82%** of Group DRO's total worst-group accuracy gain over ERM; the adversarial reweighting contributes the remaining **18%**, concentrated almost entirely on the rarest training group (waterbird/land, 56 training examples). Full details, per-group breakdowns, and the reasoning behind the ablation design are in the paper.

## Dataset

Waterbirds (Sagawa et al., 2020) combines bird photographs from [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) (Wah et al., 2011) with background scenes from the [Places](http://places2.csail.mit.edu/) dataset (Zhou et al., 2017), pasting each bird onto a land or water background so that background is correlated with label in the training set (~95%) but not in validation/test (constructed to be background-independent). This repo accesses the dataset through the [WILDS package](https://wilds.stanford.edu/), which downloads it automatically.
Group index formula (verified from WILDS' `CombinatorialGrouper`): `group = background + 2*label` → 0 = landbird/land, 1 = landbird/water, 2 = waterbird/land, 3 = waterbird/water.

## Reproducing this

All three notebooks were run on Kaggle-provided NVIDIA T4x2 instances (each 300-epoch run took roughly 7–7.5 hours). To rerun:

1. Install the dependencies in `requirements.txt` (`wilds` is the one addition on top of a standard Kaggle GPU environment; it will pull in the Waterbirds dataset automatically on first use).
2. Run `01_erm_baseline.ipynb`, then `02_group_dro.ipynb`, then `03_balanced_sampling_ablation.ipynb`, in that order. Each is self-contained.
3. Two implementation issues:
   - WILDS' group-wise evaluation path imports `torch_scatter`, an undeclared dependency that a `pip install wilds` does not pull in. This repo reimplements the group-mean reduction with `torch.scatter_add_` (ships with PyTorch).
   - WILDS' grouper builds its internal group-membership tensor on CPU and doesn't moves it, so calling it on GPU inside a training loop raises a device-mismatch error. Fix: compute group membership while metadata is still on CPU, then move only the resulting group-index tensor to the training device.

## Limitations

Every number in this repo comes from a single training run per configuration, not an average over multiple seeds, which is a consequence of compute constraints (Kaggle's free GPU quota). The original paper's Table 1 numbers are, per standard practice for this kind of benchmark, most likely averaged over multiple seeds, which is the most reasonable explanation for the gap between this repo's ERM worst-group number (68.54%) and theirs (72.6%). The rarest training group (waterbird/land) has only 56 examples, so single-run variance on that group's accuracy specifically should be expected. The decomposition result (balanced-sampling-only vs. full Group DRO) is this project's most notable contribution and is also the one most exposed to this limitation. This work also covers one benchmark (Waterbirds), one architecture (ResNet-50), at one set of hyperparameters (the paper's reference configuration). See the paper's Discussion section for the full scope discussion.

## Citation

If referencing the original Group DRO method, please cite:

```bibtex
@inproceedings{sagawa2020distributionally,
  title={Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization},
  author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

The reference implementation this repo's Group DRO loss follows is [`kohpangwei/group_DRO`](https://github.com/kohpangwei/group_DRO).

## License

MIT: see [LICENSE](LICENSE).
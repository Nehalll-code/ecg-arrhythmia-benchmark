# Transformer Encoder (Model 4): Methodology

*Owner: Nehal Verma (230953144)*

## 1. Motivation and hypothesis

The Transformer encoder (Vaswani et al., 2017) is the attention-based member of the comparison. Convolutional networks are built on local filters and recurrent networks on sequential state, whereas self-attention lets every position in the recording attend directly to every other position. In an ECG this could in principle relate beats that are far apart, for example the rhythm irregularity across a full 10 s strip, or the same lead morphology repeated across beats, without that information having to pass through many layers or time steps.

The cost of this flexibility is that attention has no built-in preference for local structure. It must learn from data what a CNN assumes by construction. The literature reports that Transformers on raw physiological signals are data-hungry and do not consistently beat well-tuned convolutional baselines (see Section A.4 of the synopsis). Our hypothesis was therefore that the Transformer would be competitive with, but not clearly better than, the CNN on PTB-XL's roughly 17,000 training records. We report whichever way this came out.

## 2. Architecture

Input is the shared tensor of shape (batch, 1000, 12): 10 s at 100 Hz, 12 leads. Output is five logits (NORM, MI, STTC, CD, HYP) passed through a sigmoid, since the task is multi-label.

| Stage | Detail |
|---|---|
| Patch embedding | `Conv1d(12 → 64, kernel 5, stride 5)`; each non-overlapping 50 ms window of all 12 leads becomes one 64-d token, giving 200 tokens |
| Class token | One learned 64-d CLS token prepended (201 tokens) |
| Positional encoding | Learned absolute positional embedding, 201 × 64 |
| Encoder | 3 pre-LayerNorm Transformer encoder layers; 4 attention heads; feed-forward width 256; GELU; dropout 0.2 |
| Head | LayerNorm on the CLS output, then `Linear(64 → 5)` |

**Why patches rather than raw samples as tokens.** Self-attention cost grows with the square of sequence length. Using each of the 1000 samples as a token would mean about 10^6 attention scores per head per layer, and a single sample carries almost no diagnostic information. Grouping 5 samples (50 ms) shortens the sequence to 201 tokens, about 25 times fewer attention scores, while a 50 ms window is still short enough to resolve the QRS complex. The embedding mixes all 12 leads at every position, so inter-lead (spatial) relationships are available from the first layer.

**Why a CLS token.** It gives a single summary vector that attends over the whole recording and is read out by a linear head, instead of averaging all tokens.

**Why pre-LayerNorm.** Normalising before each sub-layer trains more stably without a learning-rate warm-up, which we did not use.

**Size.** 167,237 trainable parameters: patch embedding 3,904; CLS 64; positional embedding 12,864; three encoder layers 149,952; final norm and head 453. This is deliberately small for a Transformer, because the training set is small. The smaller of the two other signal models (BiLSTM, 613,605) has about 3.7 times as many parameters, and the 1D-CNN (971,717) about 5.8 times as many.

## 3. Training protocol (shared with all models)

Only the model definition differs between the four architectures. Everything below was identical.

| Item | Setting |
|---|---|
| Data | PTB-XL v1.0.3, 100 Hz; 0.5 Hz second-order Butterworth high-pass (zero-phase) for baseline wander; per-lead z-score using training-fold statistics only |
| Split | Official patient-disjoint folds: 1–8 train (17,084 records), 9 validation (2,146), 10 test (2,158); records with no diagnostic superclass excluded |
| Loss | Binary cross-entropy on logits, unweighted (no class re-weighting) |
| Optimiser | AdamW, learning rate 1e-3 (constant, no scheduler), weight decay 1e-2, batch size 64, gradient-norm clipping at 1.0 |
| Stopping | Up to 50 epochs; stop when validation macro AUROC has not improved for 8 epochs; the best-validation checkpoint is restored |
| Seed | 42 |
| Augmentation | None |
| Hardware | Single NVIDIA T4 (Kaggle), about 8 s per epoch |

**Threshold selection.** Per-class decision thresholds, used only for F1, precision and recall, were chosen on the validation fold by maximising per-class F1 over a grid from 0.05 to 0.95, then applied unchanged to the test fold. AUROC is threshold-free. The test fold was scored once, after training.

**Training run.** Validation macro AUROC peaked at 0.9165 at epoch 11. Training loss continued to fall (0.259 at epoch 11, 0.233 at epoch 19) while validation AUROC fluctuated between 0.911 and 0.917 and did not improve, so training stopped at epoch 19. This is consistent with mild overfitting.

**No hyperparameter search was performed for the Transformer.** The reported model is the default configuration, trained once with one seed. The other models' owners ran tuning trials (the handoff allowed up to about six each), so the Transformer's result is probably a slight underestimate of what a tuned Transformer could reach, and this should be read as a limitation of the comparison. Untried options include a lower learning rate with warm-up, higher dropout, a larger or smaller patch size, and data augmentation.

## 4. Results

**PTB-XL test fold (n = 2,158).**

| Metric | Value |
|---|---|
| Macro AUROC [95% CI] | 0.907 [0.897, 0.915] |
| Macro F1 | 0.713 |

| Class | n (positive) | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| NORM | 963 | 0.934 | 0.817 | 0.879 | 0.846 |
| MI | 550 | 0.895 | 0.652 | 0.725 | 0.687 |
| STTC | 521 | 0.927 | 0.736 | 0.760 | 0.748 |
| CD | 496 | 0.889 | 0.818 | 0.617 | 0.703 |
| HYP | 262 | 0.888 | 0.598 | 0.561 | 0.579 |

**Comparison with the other models (same test fold).**

| Model | Macro AUROC [95% CI] | Macro F1 |
|---|---|---|
| 1D-CNN | 0.924 [0.916, 0.930] | 0.744 |
| BiLSTM | 0.924 [0.916, 0.930] | 0.738 |
| **Transformer** | **0.907 [0.897, 0.915]** | **0.713** |
| MLP (features) | 0.889 [0.878, 0.898] | 0.689 |

In a paired patient-level bootstrap of macro AUROC, the 1D-CNN and the BiLSTM each beat the Transformer in essentially every resample, and the Transformer in turn beats the MLP in essentially every resample (probabilities of 1.00). The Transformer is therefore a clear third of four, not a tie with the CNN. Its macro AUROC is also below the 0.92–0.93 that Strodthoff et al. (2021) report for residual CNNs on this task, although our protocol is not identical to theirs.

**Efficiency (per ECG, CPU, one thread, batch size 1).**

| Model | Parameters | MFLOPs | Latency (ms) |
|---|---|---|---|
| MLP (model only) | 73,989 | 0.1 | 0.18 |
| 1D-CNN | 971,717 | 178.5 | 11.43 |
| BiLSTM | 613,605 | 304.8 | 29.21 |
| **Transformer** | **167,237** | **60.8** | **8.23** |

(The MLP's end-to-end time including feature extraction is 50.1 ms.) Among the three models that read the raw signal, the Transformer is the smallest and the fastest: roughly one third of the CNN's compute and under one third of the BiLSTM's latency. It costs 0.017 macro AUROC relative to the CNN for that saving.

**External cohort (Georgia, zero-shot, four classes with enough positives).** Macro AUROC on the same four classes fell from 0.910 on PTB-XL to 0.824 on Georgia (a drop of 0.086), the largest drop of the five systems; the CNN fell by 0.063. The Transformer ranked fourth on Georgia, as it did on PTB-XL.

## 5. Discussion

**Where it is weaker.** Relative to the CNN, the AUROC gap is largest for CD (0.889 vs 0.919) and MI (0.895 vs 0.919), both classes whose diagnosis depends on the shape of the QRS complex or ST segment, and smallest for HYP (0.888 vs 0.898), where every model is weak. The Transformer's recall for CD (0.617) and HYP (0.561) is also the lowest or near-lowest of the four models.

**Possible explanations (hypotheses, not tested in this project).** (i) Self-attention has no locality or translation-equivariance prior, so it needs to learn waveform morphology from only about 17,000 recordings, whereas convolutions get it for free. (ii) Our patch embedding is a single linear (convolutional) layer over 50 ms, so fine morphology within a patch is compressed before attention sees it. (iii) The default configuration was not tuned, and a 3-layer, 64-dimension model trained with a constant learning rate and no augmentation may be under-regularised, as the growing gap between training loss and validation AUROC suggests. We did not run ablations or inspect attention maps, so we cannot say which of these, if any, accounts for the gap. A natural follow-up would be a hybrid with convolutional tokenisation, which is out of scope here.

**Takeaway.** On this task, at this data size and without tuning, attention over the raw signal did not outperform convolution or recurrence, which agrees with the cautionary literature. Its advantage is efficiency: it is the cheapest of the signal-based models. Its larger drop on the external cohort suggests it also transfers somewhat worse, though this is a single run on a single seed.

## 6. Limitations

- One training run with one seed, so seed-to-seed variance is unknown; the confidence intervals reflect test-set sampling only.
- No hyperparameter tuning for this model, unlike the others.
- Absolute-position embeddings tie the model to 10 s inputs of exactly 1000 samples.
- Latency is measured on a CPU with one thread at batch size 1 and says nothing about GPU or wearable hardware.

## References

- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*, 30, 5998–6008.
- Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE JBHI*, 25(5), 1519–1528.
- Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154.

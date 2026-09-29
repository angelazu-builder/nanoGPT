# Context-Length Order Recovery Study

## Concise technical report

> **Author:** Anqi Zu, University of Oxford  
> **Date:** September 27, 2026  
> **Model:** 4.8M-parameter character-level Transformer trained on Tiny Shakespeare  
> **Primary metric:** paired descending-minus-ascending validation BPC at evaluation horizon `T=256`

## 1. Question and original hypothesis

I asked whether the temporal order of context lengths used during training leaves a persistent effect on a language model's final performance. This was a training-dynamics question rather than a search for the best hyperparameters: every schedule used the same four context lengths—`32`, `64`, `128`, and `256`—for the same number of updates, but presented them in different temporal orders.

My original interpretation was that descending training (`256→128→64→32`) created a persistent disadvantage relative to ascending training (`32→64→128→256`). The first experiment appeared to support this: under long-context evaluation, the descending model finished about `0.17 BPC` worse. A corrected five-seed pilot then reproduced a terminal descending disadvantage. At that stage, I regarded the result as evidence that the order of context-length blocks could alter the learned model, not merely its immediate adaptation state.

The stronger hypothesis tested in the final study was therefore:

> If descending order causes persistent path dependence, a substantial positive descending-minus-ascending gap should remain after both models receive the same final long-context training condition.

The preregistered quantitative recovery prediction was that common `T=256` training would remove at least 75% of the pre-recovery gap. A positive residual above `0.03 BPC`, shared by all three paired seeds and approximately plateaued, would instead qualify only as a candidate persistent effect requiring a further mechanism experiment.

## 2. Why the original experiment could not answer the question

The first experiment was not strong enough to distinguish a persistent ordering effect from measurement and design artifacts. It used one seed, batches were not paired across schedules, evaluation targets were not consistently aligned across context horizons, and some diagnostic claims relied on a corpus probe affected by resubstitution bias. These problems meant that the dramatic result was useful for generating a question but not for establishing its answer.

The second iteration corrected much of the measurement problem. It used five paired seeds, shared initializations, matched context-specific data manifests, held-out validation text, and fixed target positions. The terminal descending deficit survived. However, the experimental design still contained a more important confound: ascending training ended at `T=256`, descending training ended at `T=32`, and the principal evaluation also used `T=256`.

Consequently, a worse descending endpoint could have at least three explanations:

1. **Persistent ordering:** earlier context-length order permanently changed the learned solution.
2. **Recent-context exposure:** each model was temporarily best adapted to the context length it had just seen.
3. **Training-stage interaction:** context length interacted with optimizer state or the late learning-rate phase.

The paired pilot showed that the endpoint difference was reproducible, but it could not identify which explanation generated it. Repeating that design with more seeds would have estimated the same confounded contrast more precisely without resolving the scientific question.

## 3. How the recovery study controlled the comparison

The recovery study used three schedules: ascending, descending, and one prespecified nonmonotonic permutation (`64→256→32→128`). Each schedule received four 500-step blocks before recovery. The confirmatory comparison was descending versus ascending; the nonmonotonic arm was exploratory.

After step 2000, all arms received 500 identical `T=256` recovery steps. This intervention matched the most recent training context and allowed the prior performance gap to be followed over time instead of measured only at a final endpoint.

Within each of three paired seeds, the study held constant:

- model architecture and initial weights;
- tokenizer, corpus, and train/validation split;
- exposure to each scheduled context length;
- 4,096 target tokens per optimizer update;
- context-specific training-data draws;
- recovery data draws at every recovery step;
- AdamW configuration, preserved optimizer state, and global learning-rate schedule;
- total training budget and recovery condition;
- validation sequences, target positions, scoring code, and checkpoints.

Only the temporal permutation of the four pre-recovery context blocks varied in the primary comparison. Because those blocks necessarily occurred at different global steps, the estimand includes the interaction of order with model state, optimizer history, and the fixed learning-rate trajectory; it is not an abstract order effect independent of training stage.

Evaluation used a fixed 16-sequence process panel at 17 checkpoints and a nested 32-sequence anchor panel at the pre- and post-recovery endpoints. Every model was evaluated at context horizons `T∈{32,64,128,256}`. The estimands, checkpoints, thresholds, plots, and decision rules were frozen before the formal results were inspected.

## 4. Core result: `+0.7570 → −0.0485 BPC`

Immediately before recovery, the paired descending-minus-ascending anchor gap at `T=256` was:

```math
\Delta^A(0)=+0.7570\ \mathrm{BPC}.
```

All three paired seeds were positive, and the 95% paired t interval was `[+0.3290,+1.1851]`. The earlier large descending deficit therefore replicated under the new matched-block design.

After 500 common `T=256` recovery steps, the same contrast was:

```math
\Delta^A(500)=-0.0485\ \mathrm{BPC}.
```

All three seed-level point estimates were now negative, but the 95% interval was wide and crossed zero: `[−0.1491,+0.0521]`. The process panel showed that the `T=256` gap fell from `+0.8097 BPC` at recovery step 0 to `+0.0719` after 10 steps, crossed zero by 50 steps, and ended at `−0.0195` after 500 steps.

The central observation is therefore:

> **The large positive descending deficit did not persist under a common long-context training condition. It rapidly reversed, while the much smaller negative residual remained unresolved.**

The preregistered recovery ratio was `R=−0.064`, corresponding descriptively to 106.4% recovery because the estimate crossed zero. This does not establish equivalence. The absolute endpoint residual, `0.0485 BPC`, remained outside the preregistered `±0.02 BPC` practical-removal band.

## 5. Why the original interpretation was weakened

If the large deficit had reflected stable damage caused by descending order, it should have remained substantially positive after the arms entered the same recovery condition. Instead, most of the difference disappeared within the first few recovery checkpoints. The pre-recovery deficit was also specific to long-context evaluation: at `T=32`, `64`, and `128`, the corresponding mean contrasts were already small and negative.

The reversal was driven mainly by improvement in the descending arm. At `T=256`, its mean BPC improved from `2.9980` to `2.2345`, whereas the ascending mean changed from `2.2410` to `2.2830`. The gap therefore did not disappear merely because the ascending reference model deteriorated toward the descending model.

Together, the speed of recovery, its concentration at the mismatched long-context horizon, and the descending arm's improvement are more consistent with recent-context adaptation than with a large persistent ordering effect. This weakens the original interpretation that descending order had placed the model in a permanently worse solution.

The result does not prove that temporal order never matters, nor does it uniquely identify recency as a mechanism. Recovery added both long-context exposure and 500 optimizer updates. It may have acted through parameter adaptation, optimizer state, representation changes, or interactions with the learning-rate schedule. The experiment rejects the large persistent-deficit interpretation more strongly than it identifies a single replacement mechanism.

## 6. What remains unknown

The most immediate unresolved question is whether the post-recovery contrast of approximately `−0.05 BPC` is reproducible. With only three paired seeds, its confidence interval includes both a moderate negative effect and a small positive effect. The study therefore cannot conclude that ascending and descending schedules are equivalent, or that the apparent sign reversal is a real smaller ordering effect.

The study also does not determine:

- which component of recovery—recent context, additional optimization, optimizer moments, or representation adaptation—caused the rapid reversal;
- whether resetting optimizer state at context transitions would change the trajectory;
- whether the result generalizes beyond this 4.8M-parameter model, character tokenizer, Tiny Shakespeare corpus, AdamW configuration, context range, or token budget;
- whether other monotonic or nonmonotonic schedules produce the same behavior;
- whether much longer matched recovery would eliminate, preserve, or amplify the small negative residual.

The defensible conclusion is narrow but informative: **the large deficit associated with descending context order did not persist under matched `T=256` recovery. Its rapid reversal is more consistent with a recent-context effect than with persistent ordering, although a smaller ordering effect cannot yet be ruled out.**

For the complete audit trail, preregistered figures, corrections, provenance, and reproduction commands, see [`TECHNICAL_REPORT_final.md`](TECHNICAL_REPORT_final.md).

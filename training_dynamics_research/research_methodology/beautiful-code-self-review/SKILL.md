---
name: beautiful-code-self-review
description: Design, refactor, or review non-trivial code by preferring clear state and data flow over accumulating patches, flags, validators, and special cases. Use when code works but feels brittle, responsibilities are mixed, lifecycle bugs recur, or the user asks whether code is elegant, overengineered, or becoming a mess.
---

# Beautiful Code Self-Review

Use this skill to make code easier to reason about, not merely shorter or more abstract. Treat elegance as an engineering property: the code should make the intended behavior obvious and invalid behavior difficult to express.

## Define the standard

Beautiful code usually has these properties:

- **Local reasoning:** a function can be understood without reconstructing distant side effects or hidden ordering requirements.
- **Explicit state:** lifecycle, mode, stage, and completion are represented directly rather than inferred from filenames, missing fields, or incidental checkpoints.
- **One source of truth:** the same fact is not encoded independently in configuration, branches, filenames, and analysis assumptions.
- **Structural invariants:** important rules follow from the design; they are not maintained only by scattered checks.
- **One-way data flow:** initialization, mutation, recording, persistence, analysis, and presentation occur in a visible order.
- **Narrow responsibilities:** each module owns a coherent decision rather than acting as an all-purpose coordinator.
- **Errors near their source:** invalid input or state fails at the boundary where it first becomes invalid.
- **Separated representations:** raw execution artifacts, validated analysis tables, scientific estimands, and presentation outputs are distinct layers.
- **Proportionate abstraction:** an abstraction removes real duplication or ambiguity; it is not added for hypothetical future use.
- **Deletion-friendly design:** a good structural change usually removes branches, repeated interpretations, or temporal coupling.

Readable names and tidy directories help, but they do not compensate for an unclear state model.

## Start with behavior and invariants

Before proposing changes, state:

1. What outcome the code must produce.
2. What must remain constant.
3. What states or transitions are valid.
4. Which failures would make the output incorrect rather than merely inconvenient.

For research code, include scientific invariants such as paired initialization, matched data, fixed evaluation targets, prespecified endpoints, and separation of debug from formal results.

Trace one representative item through the system:

```text
specification
  -> initialization
  -> execution
  -> state update
  -> persistence
  -> validation
  -> analysis
  -> presentation
```

If correctness depends on remembering an undocumented ordering between these stages, identify that as temporal coupling.

## Decide: local fix or structural change

Use a local fix when all of the following are true:

- the defect is isolated;
- the surrounding ownership and data flow are already clear;
- the fix does not introduce a new mode, lifecycle state, duplicated fact, or distant dependency;
- one focused test can protect the behavior.

Prefer a structural change when any of these are true:

- similar bugs recur in the same lifecycle;
- a function mixes initialization, execution, persistence, and interpretation;
- new behavior requires another flag in several functions;
- code infers semantic state from file existence or partial data;
- the same nested data is reinterpreted independently by many consumers;
- correctness depends on calling operations in the right order;
- validators are multiplying because invalid states remain easy to construct;
- a proposed patch adds more branches than it removes.

Do not equate refactoring with adding classes. The best structural change may be reordering operations, introducing one canonical record, separating two stages, or deleting a compatibility path.

## Prefer these structural moves

### Replace temporal coupling with an explicit lifecycle

Fragile:

```python
evaluate()
save_checkpoint()
append_log()
```

Clear:

```python
record = evaluate()
state.record(record)
state.persist_if_needed()
```

Persistence reads committed state; it never depends on a later append.

### Replace inferred modes with explicit specifications

Fragile:

```python
if two_checkpoints_exist:
    mode = "smoke"
```

Clear:

```python
RunSpec(mode="debug", stage="smoke")
RunSpec(mode="formal", stage="base")
RunSpec(mode="formal", stage="extension")
```

Debug and formal runs may share the same execution core, but their identity, output location, and acceptance rules should not be guessed afterward.

### Replace free-form nested dictionaries with one canonical record

Use a dataclass, typed mapping, table schema, or similarly lightweight representation when multiple modules rely on the same fields. Do not add a type merely to rename a dictionary; add it when it centralizes invariants or removes repeated interpretation.

### Validate once at a meaningful boundary

Validate an artifact when it enters formal analysis, then let downstream analysis and plotting consume the validated representation. Avoid making every plot rediscover missing seeds, normalize keys, infer stages, and check completeness independently.

### Separate data production from interpretation

Prefer:

```text
raw artifacts -> validated canonical tables -> estimands -> figures/report
```

Figures should not decide whether a run is formal, reconstruct experiment stages, or silently substitute missing values.

## Resist additive complexity

Before adding a flag, field, validator, wrapper, mode, or exception handler, ask:

1. What ambiguity forces this addition?
2. Can that ambiguity be removed at its source?
3. Will this addition create another legal combination of states?
4. Can an existing branch or representation be deleted instead?
5. Is this supporting a real current workflow or an imagined future one?

Warning signs include:

- boolean-flag combinations whose meanings must be memorized;
- `dict.get()` chains used to tolerate multiple schemas indefinitely;
- broad exception handlers that convert contract failures into warnings;
- comments claiming an invariant that the code does not enforce;
- completion inferred from an output file merely existing;
- analysis logic that guesses experimental identity from observed data;
- functions whose correct use requires callers to know internal operation order;
- tests that duplicate implementation branches rather than exercise public behavior.

Do not remove useful debug paths simply to make formal code look clean. Give debug runs an explicit specification and separate artifacts while reusing the same core operations.

## Implement the smallest clarifying refactor

Choose the smallest change that makes the important lifecycle or data flow explicit. A good refactor should usually achieve at least one of:

- fewer semantic branches;
- fewer representations of the same fact;
- fewer order-dependent operations;
- fewer consumers of raw internal data;
- fewer invalid states that can be constructed;
- a shorter explanation of why the code is correct.

Avoid broad rewrites when a narrow structural change solves the underlying problem. Preserve user-owned behavior, public interfaces, and unrelated work unless changing them is necessary for correctness.

## Verify behavior, not appearance

Test the invariant that motivated the change. Useful tests include:

- two paired runs begin from byte-identical initial state;
- a persisted checkpoint contains the evaluation from its own step;
- a completed artifact cannot be mistaken for partial output;
- debug artifacts cannot enter formal analysis;
- extension logic reaches a hard-cap terminal state rather than extending again;
- canonical tables have the expected keys, row counts, and finite values;
- repeated analysis of the same artifacts is deterministic;
- removing a required record causes a loud boundary failure.

Avoid tests that only assert internal helper calls, repeat the production `if/elif` tree, or lock in incidental formatting.

## Perform a fresh-reader self-review

After implementation, stop thinking like the author and inspect the code in this order:

1. **Entry point:** Can a reader identify the complete workflow quickly?
2. **State model:** Are mode, stage, progress, and completion explicit?
3. **Initialization:** Are seeds, configuration, and external inputs fixed before dependent objects are created?
4. **Mutation order:** Is state updated before it is persisted or consumed?
5. **Failure behavior:** Do validity failures stop formal execution rather than become warnings?
6. **Data contract:** Is there one canonical representation between execution and analysis?
7. **Boundaries:** Are debug, formal, base, extension, raw, analyzed, and presented outputs distinguishable?
8. **Branch pressure:** Did the change reduce or increase the number of semantic paths?
9. **Test quality:** Do tests protect outcomes and invariants rather than mirror implementation?
10. **Explainability:** Can correctness be explained in a short causal chain without saying “provided callers remember to…”?

Then inspect the diff and ask:

- What became simpler?
- What new concept was introduced?
- Which old concept or branch did it replace?
- Is the new concept earning its cost?
- Could the same result be achieved by reordering or deleting code?

If a refactor adds vocabulary and files without removing ambiguity, it is probably architecture theater rather than improvement.

## Give an honest verdict

When reviewing code, avoid both politeness inflation and theatrical insults. Classify it as:

- **Elegant:** invariants and lifecycle are clear; extension is straightforward.
- **Serviceable:** understandable and safe enough, with localized debt.
- **Brittle prototype:** works, but correctness relies on ordering, implicit state, or duplicated interpretation.
- **Patch-accumulating system:** each change adds flags and compatibility branches; structural repair is due.
- **Unmaintainable:** behavior cannot be changed confidently without broad unintended effects.

Support the verdict with concrete data-flow or state-model evidence. A clean folder tree is not proof of elegant code, and an imperfect research script is not automatically a “mess.”

## Produce an actionable response

Return:

1. the honest verdict and the evidence for it;
2. the strongest parts worth preserving;
3. the central structural problem, not a long undifferentiated bug list;
4. whether to patch locally or refactor structurally, with the reason;
5. the smallest proposed target architecture or data flow;
6. what should be deleted, merged, or made explicit;
7. the invariants and behavioral tests required before completion;
8. remaining limitations or deliberately retained shortcuts.

Separate correctness-critical changes from aesthetic improvements. Under time or compute constraints, fix the state model and scientific validity first; do not spend the budget polishing low-risk abstractions.

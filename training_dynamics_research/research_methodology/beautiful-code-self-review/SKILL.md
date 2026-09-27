---
name: beautiful-code-self-review
description: Design, refactor, or review non-trivial code by reducing entangled state, leaky interfaces, speculative abstraction, and patch accumulation. Use when code works but feels brittle, lifecycle bugs recur, responsibilities are mixed, or the user asks whether code is elegant, overengineered, or becoming a mess.
---

# Beautiful Code Self-Review

Use this skill to make code easier to understand, change, test, and trust. Do not equate elegance with short functions, many layers, zero duplication, strict style rules, or impressive abstractions.

Judge the design in its actual context. A contest algorithm, research prototype, product service, and safety-critical database should not pay the same abstraction or verification costs.

## The standard

Beautiful code tends to have:

- **Local reasoning:** understand a unit without reconstructing distant mutations or hidden call order.
- **Low entanglement:** concerns that change independently are not interwoven in state, time, or control flow.
- **Deep modules:** a small, stable interface hides substantial implementation knowledge and provides meaningful capability.
- **Cohesive ownership:** each important fact or design decision has one natural home.
- **Explicit lifecycle:** mode, stage, progress, completion, and legal transitions are represented directly.
- **One-way data flow:** initialization, execution, state update, persistence, validation, analysis, and presentation have a visible order.
- **Errors near their source:** invalid state is rejected at the boundary where it first becomes invalid.
- **Proportionate machinery:** every abstraction, dependency, configuration option, and validation layer earns its cost.
- **Changeability:** likely changes stay local instead of propagating through callers and unrelated modules.
- **Evidence:** tests and measurements support the properties the code claims to have.

Readable names and tidy folders are useful, but they cannot rescue a leaky state model.

## Start from behavior and knowledge

Before reviewing or changing code, state:

1. What outcome must be produced?
2. What invariants make that outcome trustworthy?
3. What states and transitions are legal?
4. Which design knowledge should have a single owner?
5. What is expected to change independently?
6. What failure cost and performance constraints apply?

For research code, include scientific invariants such as paired initialization, matched data, fixed targets, prespecified endpoints, and isolation of debug artifacts from formal evidence.

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

Mark every place where a reader must know something not visible at the interface. These hidden dependencies and obscured facts are stronger evidence of complexity than line count.

## Diagnose the shape of the complexity

Look for four forms of friction:

### Change amplification

One conceptual change requires edits in many places. Common causes are duplicated knowledge, leaky abstractions, and multiple representations of the same state.

### Cognitive load

A reader must hold too many concepts, flags, branches, or cross-file assumptions at once. A large cohesive function can be easier than several pass-through helpers; count concepts and dependencies, not methods.

### Unknown unknowns

It is unclear what else may break. Warning signs include ambient global state, mode inference from files, broad exception handling, and schemas that vary by caller.

### Temporal coupling

Correctness depends on operations happening in an undocumented order, such as saving before or after a log mutation. Make the lifecycle explicit or combine the knowledge in one owner.

## Decide: patch or restructure

Use a local patch when:

- the defect is isolated;
- ownership and data flow are already clear;
- the change adds no new semantic mode or duplicated fact;
- one focused behavioral test protects it.

Prefer a structural change when:

- similar bugs recur in the same lifecycle;
- one function mixes initialization, execution, persistence, and interpretation;
- a feature requires the same new flag across several layers;
- semantic state is inferred from filenames, missing fields, or observed outputs;
- many consumers independently reinterpret the same nested data;
- validators multiply because invalid states remain easy to construct;
- the proposed patch adds more legal combinations or branches than it removes.

Do not equate restructuring with adding classes. Reordering operations, unifying a record, separating stages, deleting a compatibility path, or moving knowledge behind one interface may be enough.

## Choose a clarifying structural move

### Commit state before persistence

Prefer:

```python
record = evaluate()
state.record(record)
state.persist_if_needed()
```

over an ordering contract such as `evaluate -> save -> append` that callers must remember.

### Represent semantic modes explicitly

Prefer:

```python
RunSpec(mode="debug", stage="smoke")
RunSpec(mode="formal", stage="base")
RunSpec(mode="formal", stage="extension")
```

over guessing identity from the number of checkpoints or existence of files. Debug and formal paths may share the same execution core while keeping distinct specifications and artifacts.

### Build deep rather than numerous shallow modules

A module should hide a meaningful decision and make common use simpler. Avoid pass-through wrappers, classes that merely rename a dictionary, and helpers whose interfaces expose nearly all of their implementation.

Do not split a cohesive function solely because it is long. Split when the new boundary hides knowledge, reduces coupling, or creates a testable concept.

### Establish one canonical data boundary

Prefer:

```text
raw artifacts -> validated canonical records/tables -> estimands -> figures/report
```

Validate once when artifacts enter formal analysis. Downstream figures should not rediscover missing seeds, normalize schemas, infer stages, or silently substitute values.

### Distinguish duplicated text from duplicated knowledge

Similar-looking code is not automatically the same concept. Remove duplication when copies encode one shared rule that must change together. Keep duplication temporarily when the pieces belong to different domains or are likely to evolve independently.

Do not trade obvious, changeable code for a compressed abstraction whose parameters and conditionals recreate every original special case.

## Control abstraction and review cost

Before adding a flag, wrapper, type, validator, dependency, or configuration option, ask:

1. What ambiguity or dependency does this remove?
2. Which existing branch, representation, or concept becomes unnecessary?
3. Does the interface become simpler for its callers?
4. Is this serving a current need or a speculative future one?
5. Will the next likely change become more local?

If nothing can be deleted or hidden, the new abstraction may be architecture theater.

Make changes reviewable:

- keep each change conceptually self-contained;
- separate behavior-preserving refactors from feature changes when mixing them obscures the diff;
- include the relevant tests with the change;
- preserve a working state after each committed step;
- prefer continuous improvement over blocking useful work for subjective perfection.

Classify review feedback as correctness-critical, design debt, or optional polish. Do not present personal style preferences as correctness requirements.

## Comments and names

Names should communicate domain meaning, units, and distinctions such as index versus count. Comments should preserve information the code cannot express well:

- why a decision was made;
- the invariant or contract;
- non-obvious constraints and trade-offs;
- the reason an apparently simpler alternative is unsafe.

Do not use comments to narrate confusing code that can instead be simplified. Keep comments synchronized with behavior, and delete obsolete explanations and TODOs.

## Verify the mental model

Test the invariant that motivated the design. Prefer tests that would fail under the suspected defect:

- paired runs begin from identical initial state;
- a checkpoint contains the evaluation from its own step;
- debug artifacts cannot enter formal analysis;
- a terminal stage cannot transition back into extension;
- canonical tables have expected identities, row counts, and finite values;
- removing a required record produces a loud boundary failure;
- repeated analysis of the same artifacts is deterministic.

For algorithms or transformations with a slow, obvious implementation, use differential or stress testing:

```text
optimized implementation(input) == trusted reference(input)
```

over many generated and adversarial cases. Record assumptions, complexity, and testing status close to reusable algorithmic code.

Use assertions for programmer errors and invariants when appropriate to the failure cost. Do not copy safety-critical numerical rules, assertion quotas, or hard size limits into ordinary projects without justification. Assertions supplement a precise mental model; they do not replace it.

Tests are production code too. Reject tests that duplicate the implementation's branch tree, pass when the behavior is broken, or require more reasoning than the code they protect.

## Fresh-reader self-review

After implementation, inspect the whole affected path rather than only the diff:

1. **Entry point:** Is the workflow discoverable?
2. **Interfaces:** Are modules deep enough to justify their existence?
3. **State:** Are mode, stage, progress, and completion explicit?
4. **Initialization:** Are inputs and seeds fixed before dependent objects are created?
5. **Mutation:** Is state committed before persistence or consumption?
6. **Failures:** Do correctness failures stop at the right boundary?
7. **Data contract:** Is execution interpreted once rather than by every consumer?
8. **Coupling:** What knowledge leaks across module boundaries?
9. **Branch pressure:** Did the change reduce semantic paths or merely redistribute them?
10. **Tests:** Would they catch the defect, including invalid and boundary cases?
11. **Performance:** Is important algorithmic or resource cost visible and measured?
12. **Explainability:** Can correctness be explained without “provided callers remember to…”?

Review the diff again and ask:

- What became simpler for the next reader?
- What new concept was introduced, and which old complexity did it replace?
- Did an abstraction localize future change or merely reduce line count?
- Could reordering, deletion, or a deeper existing module achieve the same result?
- Are unrelated formatting and cleanup obscuring the semantic change?

## Give an honest verdict

Classify the code as:

- **Elegant:** important knowledge is hidden behind clear interfaces; lifecycle and invariants are visible.
- **Serviceable:** understandable and safe enough, with localized debt.
- **Brittle prototype:** works, but correctness relies on ordering, implicit state, or duplicated interpretation.
- **Patch-accumulating system:** changes add flags and compatibility paths faster than they remove ambiguity.
- **Unmaintainable:** ordinary changes cannot be made confidently without broad unintended effects.

Support the verdict with state-flow, dependency, and change-amplification evidence. A clean directory tree is not proof of good design; a terse contest solution is not automatically maintainable; an imperfect research script is not automatically a mess.

## Produce an actionable review

Return:

1. the verdict and strongest evidence;
2. what should be preserved;
3. the central structural problem rather than an undifferentiated bug list;
4. patch versus restructure, with the reason;
5. the smallest target data flow or ownership model;
6. what becomes explicit and what can be deleted;
7. the invariants and behavioral tests required;
8. deliberately retained shortcuts and remaining limitations.

Separate correctness-critical work from design debt and optional polish. Under tight time or compute constraints, repair the state model and validity first; do not spend the budget on low-risk aesthetic uniformity.

# Documentation standard

This document defines the project's writing, organization, and documentation review rules. It is the canonical standard for guides, reports, process records, model cards, configuration comments, and generated documentation. Apply it when creating or updating those surfaces.

## Language and author voice

Write documentation and configuration comments in natural, precise English, including records in `docs/process/`. Preserve identifiers, commands, paths, original quotations, and multilingual dataset or API examples where their exact content matters.

Write from the project author's perspective. State project behavior, findings, decisions, and requirements directly. Describe an actual API user when relevant, but remove assistant-to-owner narration such as “the user requested,” “the owner authorized,” “as instructed,” and “awaiting user confirmation.” A reader should understand the document without access to a conversation.

Use familiar words and accurate technical terms. Open with the purpose, result, or action the reader needs. Use paragraphs for explanation and tables for comparisons. Avoid promotional adjectives, artificial contrasts, repeated conclusions, decorative formatting, and em dashes. Technical reports use consistent, neutral labels and preserve the severity of observed failures.

## One responsibility per document

Choose the document's role before adding content:

| Surface | Responsibility |
|---|---|
| Project README | Explain the project, provide a first run, and link to guides |
| Documentation index | Route readers by task without repeating the guides |
| Usage guide | Describe supported behavior, requirements, commands, and recovery steps |
| Dated report in `docs/reports/` | Identify the evaluated artifact, measurement conditions, verified results, observed failures, and the decision supported by those results |
| Record in `docs/process/` | Preserve experiment plans, investigations, training attempts, selection history, amendments, and follow-up proposals |
| Standalone model card | Explain the distributed artifact, how to run it, and where to find its report and licensing |

Keep unfinished plans, training progress, abandoned approaches, and execution diaries in process records. Formal guides and final reports link to the specific record when that context is needed. A failed evaluation belongs in a report when it is a measured result. A proposal to repair it belongs in the process record.

Group process records by experiment or purpose in their index. Add every new record to that index and connect it to its predecessor or relevant guide. Keep a dated record's identity clear after later experiments supersede it.

## One detailed source per fact

Each fact has one canonical home. Put the complete rule, table, result, or procedure there. Other pages provide the context needed for their own task and a specific link. Repeating the details and then linking to them still creates a duplicate.

Use progressive disclosure: overview, task guide, then detailed evidence or implementation. Keep detailed settings and result tables out of indexes and introductions. Split a page when it starts serving unrelated tasks, leaving a short purpose statement and link in its former location.

Code owns defaults, validation limits, supported options, and file-selection logic. Link to the file and name the responsible function, symbol, or data key. A repository or directory link alone is insufficient for a specific rule. Registered manifests and result files own experiment counts, hashes, and raw measurements. A dated report may present the selected results needed to assess its conclusion.

## Accurate claims and durable state

Describe implemented behavior from code and measured results from the corresponding artifacts. Keep planned, implemented, executed, verified, exported, and published states distinct. A completed training run alone does not establish that its candidate passed evaluation or release checks.

Durable guides describe how to obtain the current answer. Avoid hand-maintained snapshots of versions, counts, running jobs, deployment state, or release blockers. Put dated observations in their experiment or release record. Identify a historical candidate by its phase, run, or artifact reference instead of “latest” or “current.”

Reports must identify the model and evaluation cohort. Distinguish the original checkpoint before project fine-tuning from a pretraining-only Base model and from an intermediate parent adapter. Compare models on identical requests and state relevant calibration, language coverage, source grouping, and timing scope. Keep development, exposed regression, and independent acceptance results labeled by their actual roles.

Preserve errors, failed criteria, uncertainty, and conditions that affect interpretation. Correct stale wording using evidence, without changing recorded numerical results or implying that later work was already complete at the time of an earlier report.

## Direct statements without disclaimer prose

Remove generic disclaimers and habitual defensive endings such as “this does not prove,” “this is not a claim,” and “training completion does not mean the issue is fixed.” State the actual result and its necessary conditions directly.

| Avoid | Write instead |
|---|---|
| “This record does not claim the defect is fixed.” | “This record defines the experiment data and recipe.” |
| “The user authorized another run.” | Record the chosen recipe and stopping rule in the experiment plan. |
| “This is not an independent benchmark.” | “This cohort is exposed regression data, grouped by original source.” |
| “The latest model is still blocked.” | “The Phase 4 candidate failed the complete-field condition check.” |

Removing disclaimers must preserve factual limits, input requirements, measured failures, and uncertainty that changes the reader's decision. Express those facts in the claim itself. Keep license texts and third-party attribution complete, including their legal warranty provisions. Licensing scope is defined only in [licensing and third-party notices](../THIRD_PARTY_NOTICES.md).

## Generators, copies, and frozen evidence

When editing generated documentation, update the responsible template so the change survives regeneration. Verify its rendered text against the reviewed document. Avoid rerunning a report pipeline over manually reviewed conclusions merely to change wording.

Standalone exports may contain documentation copies because they ship independently. Synchronize only documents and templates that correspond to that package's runtime and scope. An older package must not inherit instructions for unavailable modules or features. Update affected checksums after assembling a mutable export.

Frozen experiment materials, archived source snapshots, uploaded run packages, and their recorded hashes remain intact. Link to them as evidence. If a registered protocol needs amendment, preserve its original registration and record the new revision explicitly. Do not translate or rewrite frozen evidence in place.

## Review before completion

Check the parts affected by the change:

- Verify commands, code paths, symbols, flags, and behavioral descriptions against the implementation.
- Check local links, translated or renamed heading anchors, index coverage, and references from report templates.
- Scan for unintended Chinese prose, assistant dialogue, generic disclaimers, duplicated facts, and obsolete progress statements.
- Compare numerical results, model identities, hashes, and relevant measurement conditions with the original evidence.
- Check generated text and standalone copies where applicable, then verify affected checksums.
- Preserve concurrent edits, frozen evidence, and unrelated training or inference behavior. Use documentation and focused template checks for prose changes.

Revise this standard when a documentation rule changes. Link to it from contributor-facing pages instead of copying its rules into each guide.

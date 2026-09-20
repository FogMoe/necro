# Phase 2 supplementary source isolation audit

Before reading final predictions, the audit examined source relationships beyond complete requests. The initial `experiment-v3` isolated translated questions, passages, and question IDs, but a retrieval question ID does not identify an independent document, and a complete PAWS sentence pair does not identify independent source sentences.

In the earlier slices, training and final retrieval questions shared 18 correct documents, or 49 documents when negative candidates were included. Another 3 normalized sentences crossed calibration or development and test boundaries. Training for the two loss comparisons was retained, while evaluation was reorganized by structural sources without using test performance to remove questions. At that point, neither final-set predictions nor Jev predictions on the new questions had been run.

`source_isolation.py` links translations and different PAWS pairs through their original English sentences, XNLI records through English premises, and retrieval questions through all candidate documents. Connected components of shared sources define statistical groups. Deduplication follows source relationships rather than labels or model answers.

The revised registration is in `data/phase2/strict-v1`:

| Role | Questions | Connected source groups |
|---|---:|---:|
| Source audit records, including duplicates | 8,712 records / 5,800 distinct requests | 2,830 |
| Development | 707 | 305 |
| Calibration | 649 | 272 |
| Sealed test | 1,171 | 503 |

Development excluded 19 retrieval questions whose sources had appeared in training, and calibration excluded 26. The sealed set excluded 4 paraphrase questions with overlapping source sentences. Its 80 retrieval questions were resampled with a predefined seed, using correct and negative candidate documents absent from training, original development, and original calibration materials. Official test queries, length limits, randomized candidate counts, and the treatment of unjudged negatives were retained.

Existing predictions were reused and reaggregated for development questions whose requests and labels were unchanged, retaining original reports. Task-macro accuracy was 74.37% for round4 and 85.15% for the first answer-ce run. The inference regression from 85% to 81% remained after source cleanup.

Source audit record counts differ from cumulative training presentations: the same expanded data was used for both round3 and round4. Training volume must be totaled separately along the final adapter's parent-weight chain.

The additional paired perturbations are registered in `data/phase2/robustness-v5`: 140 questions derived from final-set questions, analyzed through their pairing with the originals. Before predictions were read, v5 replaced v4's generic meta-question negation with direct task-specific condition negation to avoid combining conflicting answer instructions. Earlier snapshots are retained. The refinement dataset is registered as `data/phase2/refinement-v2`, with 2,800 records covering multi-field numeric counterfactual pairs, original-task replay, and natural training examples unused in Phase 2. Training records determine whether it was actually used.

Performance thresholds and diminishing-return stopping rules remain those in the [acceptance protocol](phase2-acceptance.md). Source cleanup did not relax them. Original snapshots are retained for audit, while subsequent selection, calibration, and final comparisons use the strict-source version.

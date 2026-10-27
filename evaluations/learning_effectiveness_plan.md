# TISW learning-effectiveness evaluation plan

This plan is intentionally separate from automated response-quality scores.
The project must not claim that TISW improves learning until learner data has
been collected and analysed.

## Protocol

1. Recruit learners with consent and record prior experience, subject, and
   language preference without storing unnecessary identity data.
2. Randomly assign a concept and a response condition (Normal or TISW) while
   keeping the question, model, and allowed time comparable.
3. Give a **pre-test** with concept definition, explanation, and one transfer
   item. Score using a fixed answer key and blinded reviewers.
4. Give the learner one or more tutor interactions. Preserve the transcript
   and record latency, errors, and whether the learner asked for clarification.
5. Give an **immediate post-test** with parallel but non-identical items.
6. Give a **delayed retention test** approximately seven days later without
   showing the original answer first.
7. Give a **transfer task** that requires applying the concept in an unfamiliar
   situation. Score the reasoning, not just the final answer.

## Measures

- Pre/post gain: `(post-test score - pre-test score) / (maximum - pre-test score)`.
- Delayed retention: delayed-test score and loss from immediate post-test.
- Transfer: rubric score on the new situation, reported separately from recall.
- Report confidence intervals and sample size, missingness, and attrition.
- Compare conditions with the same concept mix and adjust for baseline score;
  do not interpret a longer response as a learning gain.

## Required safeguards

- Blind reviewers to mode where practical and measure inter-rater agreement.
- Pre-register the primary outcomes and exclusion rules before looking at
  condition differences.
- Keep this learner dataset separate from `evaluations/dataset.py` and never
  place personal data, API keys, or private transcripts in the repository.

The current repository contains the protocol only. No learner outcomes are
measured by the automated runner.

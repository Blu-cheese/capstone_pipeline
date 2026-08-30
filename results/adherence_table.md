# Taxonomy Adherence: Freeform vs Schema-Constrained Extraction

Model: `gemini-2.5-flash-lite` via the Gemini OpenAI-compatible endpoint.
Adherence = share of extracted relations that are one of the 20 taxonomy types.
Measured on the saved triple files (post entity-resolution) for all three runs.

| Run | meeting_001 | meeting_002 | Combined | Entity-type violations |
|---|---|---|---|---|
| Original (April, freeform) | 1/26 (4%) | 16/18 (89%) | **17/44 (38.6%)** | 0/88 |
| Freeform (re-run) | 4/28 (14%) | 12/27 (44%) | **16/55 (29.1%)** | 0/110 |
| Constrained (schema enums) | 36/36 (100%) | 49/49 (100%) | **85/85 (100.0%)** | 0/170 |

## Off-taxonomy relations invented

**Original (April, freeform)** — 23 distinct:  `allocating`, `by`, `campus visit on`, `completed before`, `deadline for`, `ensure`, `extending`, `handle`, `introducing`, `keeping at`, `need revision`, `needs_updating`, `proposing`, `recommended`, `reports`, `requiring`, `scheduled for`, `should be notified`, `supported by`, `will coordinate`, `will notify`, `will update`, `worked well in`

**Freeform (re-run)** — 33 distinct:  `allocating`, `assigned`, `attended`, `attended_by`, `by`, `campus_visit_on`, `cleared_technical_round`, `completed_before`, `confirmed_for`, `estimated_completion_time`, `extending`, `final_results_expected`, `handle`, `introducing`, `keeping_at`, `need_revision`, `needs_to_arrange`, `needs_to_book`, `needs_updating`, `occurred_in`, `occurred_on`, `please_ensure`, `proposing`, `recommended`, `reported`, `reports`, `requiring`, `should_be_notified`, `supported_by`, `will_coordinate`, `will_notify`, `will_update`, `worked_well_in`

**Constrained (schema enums)** — 0 distinct:  _none_

## Extraction yield

| Run | Triples (001) | Triples (002) | Total |
|---|---|---|---|
| Original (April, freeform) | 26 | 18 | 44 |
| Freeform (re-run) | 28 | 27 | 55 |
| Constrained (schema enums) | 36 | 49 | 85 |

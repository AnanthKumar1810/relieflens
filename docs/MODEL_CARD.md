# Model card

## Intended use

Assist analysts in triaging paired post-disaster satellite imagery for further human review.

## Not intended for

Autonomous emergency response, safety-critical dispatch, legal determinations, or decisions without human review.

## Input

Co-registered RGB pre-disaster and post-disaster satellite image tiles.

## Output

Building probability map, semantic damage class map, calibrated class probabilities after validation temperature scaling, and tile-level review priority.

## Known risks

False positives from shadows, water, vegetation and co-registration errors; missed small buildings; class imbalance; disaster-domain shift.

## Evaluation requirement

Use event-disjoint test disasters and report per-class results, calibration, and top-k triage recall. Do not treat synthetic test fixtures as model performance evidence.

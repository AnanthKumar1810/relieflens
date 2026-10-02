# xBD data contract

The implementation expects the original xBD-style pairing:

- `<event>_<id>_pre_disaster.png`
- `<event>_<id>_post_disaster.png`
- `<event>_<id>_pre_disaster.json`
- `<event>_<id>_post_disaster.json`

The post label should contain pixel-space building polygons in `features.xy`. Each building feature contains `properties.uid`, `properties.feature_type`, `properties.subtype`, and a WKT polygon. The `subtype` values used by this project are `no-damage`, `minor-damage`, `major-damage`, and `destroyed`; `un-classified` is ignored for damage loss/metrics.

## Data split policy

Do not split individual tiles from the same disaster between train and test. Put the disaster event names explicitly into `configs/default.yaml`.

## Practical subset recipe

For a first serious experiment, choose four disaster events with enough paired labels and use two for training, one validation, and one unseen test event. Expand later without changing the evaluation protocol.

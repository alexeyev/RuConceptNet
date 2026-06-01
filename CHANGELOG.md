# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

### Added
- Edge **weights** (ConceptNet assertion confidence) are now parsed from the
  raw dump and stored in the graph. `get_targets`, `get_sources`, and
  `check_pair` accept `with_weights=True` to return `{relation: weight}`
  mappings instead of plain relation sets/lists.
- Faster `get_sources`: column look-ups now use a lazily-built, cached CSC
  matrix (~100x faster on representative data) without changing the stored
  data file.

### Changed
- `Sparse3DTensor` cells now hold the edge weight instead of a constant `1.0`;
  `row_nz`/`col_nz` return `(indices, depths, weights)` and `rowcol_nz` returns
  `(depths, weights)`.
- When several assertions share the same `(source, target, relation)`, the
  maximum weight is kept.

### Compatibility
- The public `ConceptNet` API is backward compatible: without `with_weights`
  the return shapes are unchanged.
- Data files built before this release still load; every edge simply reports a
  weight of `1.0`. Rebuild the bundled data with `prepare_data.sh` to populate
  real weights.

## [0.0.3]

### Changed
- Migrated packaging to PEP 621 (`pyproject.toml`); removed `setup.py`/`distutils`.
- Dropped end-of-life Python 3.9; minimum supported version is now 3.10.
- Modernised resource loading to `importlib.resources.files()` (works on 3.13+).
- Added `py.typed`, a pre-commit config, refreshed CI (matrix 3.10–3.14), and
  fixed a `.gitignore` rule that was ignoring `requirements*.txt`.

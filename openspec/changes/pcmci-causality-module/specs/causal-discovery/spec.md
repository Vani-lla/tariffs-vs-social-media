# causal-discovery Specification Delta

## ADDED Requirements

### Requirement: Run causal discovery with PCMCI+ by default
System MUST expose a configurable causal discovery module based on Tigramite that uses PCMCI+ as the default mode to support contemporaneous effects (`tau=0`).

#### Scenario: Default configuration uses PCMCI+
- **GIVEN** a valid stationary multivariate daily dataframe
- **WHEN** the analyzer is created with default configuration
- **THEN** `use_pcmciplus` is `True`
- **AND** `tau_min` equals `0`
- **AND** the run path calls PCMCI+.

#### Scenario: Classic PCMCI mode is supported
- **GIVEN** a valid stationary multivariate daily dataframe
- **AND** configuration sets `use_pcmciplus=False`
- **WHEN** the analyzer is executed
- **THEN** the run path calls classical PCMCI
- **AND** `tau_min` equals `1`.

### Requirement: Return significant causal links in tabular form
System MUST parse Tigramite result matrices and return only statistically significant links sorted by absolute MCI strength.

#### Scenario: Significant links are returned and sorted
- **GIVEN** `val_matrix` and `p_matrix` from an analysis run
- **WHEN** links with `p_value <= alpha_level` are parsed
- **THEN** output contains columns `cause_var`, `effect_var`, `lag_tau`, `mci_val`, `p_value`
- **AND** rows are sorted descending by `|mci_val|`.

#### Scenario: No significant links found
- **GIVEN** a valid analysis result with no links below `alpha_level`
- **WHEN** results are parsed
- **THEN** output is an empty dataframe with the expected columns and no exception is raised.

### Requirement: Map daily merged columns to standardized causal variables
System MUST provide an explicit adapter that maps columns from `clean_out/merged/daily.csv` into standardized variables required by the causal analysis API.

#### Scenario: Default mapping produces standardized columns
- **GIVEN** an input dataframe that includes mapped source columns from `daily.csv`
- **WHEN** the default adapter mapping is applied
- **THEN** output contains `media_framing`, `delta_vix`, `sp500_return`, `delta_yield_10y`.

#### Scenario: Missing source columns are rejected
- **GIVEN** an input dataframe missing at least one mapped source column
- **WHEN** adapter mapping is applied
- **THEN** the adapter raises a clear validation error.

### Requirement: Optionally save a causal graph visualization
System MUST provide an optional graph export method that writes a causal graph to a PNG file.

#### Scenario: Graph is exported to PNG
- **GIVEN** analysis results containing graph and score matrices
- **WHEN** graph export is requested with an output path
- **THEN** a PNG file is written at that path.


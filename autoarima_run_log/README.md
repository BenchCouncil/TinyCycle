# AutoARIMA experiment logs

| Folder | Contents |
| --- | --- |
| `01-baseline-p719/` | 720-step baseline with an AR order cap of 719 |
| `02-protocol-alignment/` | Initial upstream comparison, retry after the constant-window discrepancy, and recheck |
| `03-validation-search/` | Initial candidate fitting, validation and test run progress, and candidate failure records |
| `04-validation-recheck/` | Rescoring with the upstream constant-window behavior and candidate failure records |
| `05-table1-four-core-timing/` | Fixed-configuration refit and test timing used in Table 1 |

The logs retain the recorded timings, metrics, warnings, and errors. Local and remote paths are replaced with relative references or placeholders. Absolute start times are omitted. Exit-code files are included where captured by the original run. The original logs are kept in their respective run directories.

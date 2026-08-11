# Variables required from the authorized NEDIS extract

The public repository includes variable names and analytic use, but no NEDIS records or code-frequency outputs. Definitions below follow the customized extract's variable-description worksheet.

| Variable | Source description | Analytic use |
|---|---|---|
| `ptmiemar` | ED location (municipality code) | Regional descriptive grouping |
| `ptmiemcl` | ED facility category | Eligibility and ED-level analyses |
| `ptmiemnm` | ED identifier token | Stable-facility, continuity, and clustered analyses |
| `ptmiindt` | ED arrival date | Study period and calendar week |
| `ptmiintm` | ED arrival time | Arrival hour and ED length of stay |
| `ptmibrtd` | Age group | Adult eligibility and case mix |
| `ptmisexx` | Sex | Case mix and adjusted analyses |
| `ptmiiukd` | Insurance type | Descriptive case mix |
| `ptmiinrt` | Arrival route | Descriptive and adjusted analyses |
| `ptmiinmn` | Arrival mode | Descriptive and adjusted analyses |
| `ptmikts1` | Initial KTAS level | Severity description and adjusted analyses |
| `ptmiemrt` | ED disposition | Transfer, admission, ICU, and ED-death outcomes |
| `ptmihsrt` | Admission route | ICU-definition sensitivity analysis |
| `ptmiotdt` | ED exit date | ED length of stay |
| `ptmiottm` | ED exit time | ED length of stay |
| `ptmidcrt` | Postadmission outcome | Index-hospital mortality outcome |
| `ptmidctp` | Receiving-facility type | Transfer-destination outcomes |
| `dgotdiag01`–`dgotdiag20` | ED discharge diagnosis codes | AMI and stroke strata |
| `dgotdggb01`–`dgotdggb20` | ED discharge diagnosis type | Identification of principal diagnosis |

The import step also profiles several source fields for quality control. No profile output is committed.


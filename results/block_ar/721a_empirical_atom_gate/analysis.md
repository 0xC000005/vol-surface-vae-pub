# 721a Empirical Atom-Gate Diagnostic

| scope | variant | pass | mean KS | AAA KS | BBB KS | AAA zero gen/gt | BBB zero gen/gt | factor corr abs ratio | conditional |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor` | `identity` | `11/13` | `0.107` | `0.287` | `0.293` | `0.000/0.474` | `0.000/0.383` | `0.616` | `4.54` |
| `anchor` | `global_atom` | `11/13` | `0.101` | `0.226` | `0.276` | `0.435/0.474` | `0.266/0.383` | `0.562` | `4.54` |
| `anchor` | `history_bin_atom` | `11/13` | `0.101` | `0.225` | `0.275` | `0.454/0.474` | `0.305/0.383` | `0.562` | `4.54` |
| `joint` | `identity` | `11/13` | `0.101` | `0.239` | `0.260` | `0.000/0.474` | `0.000/0.383` | `0.355` | `4.23` |
| `joint` | `global_atom` | `11/13` | `0.100` | `0.211` | `0.274` | `0.434/0.474` | `0.265/0.383` | `0.330` | `4.23` |
| `joint` | `history_bin_atom` | `11/13` | `0.099` | `0.210` | `0.272` | `0.454/0.474` | `0.304/0.383` | `0.331` | `4.23` |

## Selected Atom Channels

- `anchor` selected: `['factor:aaa_oas', 'factor:bbb_oas']`
- `anchor` global p: `{'factor:aaa_oas': 0.43533533811569214, 'factor:bbb_oas': 0.26549482345581055}`
- `anchor` history-bin mean p: `{'factor:aaa_oas': 0.4539157450199127, 'factor:bbb_oas': 0.3046981692314148}`
- `joint` selected: `['factor:aaa_oas', 'factor:bbb_oas']`
- `joint` global p: `{'factor:aaa_oas': 0.43533533811569214, 'factor:bbb_oas': 0.26549482345581055}`
- `joint` history-bin mean p: `{'factor:aaa_oas': 0.4539157450199127, 'factor:bbb_oas': 0.3046981692314148}`

## Decision

If empirical atom gates pass BBB while preserving correlation, implement a learned hurdle/sticky gate. If they do not, the nonzero continuous path distribution must be repaired before adding a gate to the production model.

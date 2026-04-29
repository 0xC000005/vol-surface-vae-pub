# 721a Empirical Atom-Gate Diagnostic

| scope | variant | pass | mean KS | AAA KS | BBB KS | AAA zero gen/gt | BBB zero gen/gt | factor corr abs ratio | conditional |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor` | `identity` | `9/13` | `0.168` | `0.439` | `0.519` | `0.000/0.474` | `0.000/0.383` | `0.728` | `3.62` |
| `anchor` | `global_atom` | `9/13` | `0.138` | `0.232` | `0.336` | `0.436/0.474` | `0.266/0.383` | `0.637` | `3.62` |
| `anchor` | `history_bin_atom` | `9/13` | `0.138` | `0.231` | `0.335` | `0.454/0.474` | `0.305/0.383` | `0.634` | `3.62` |
| `joint` | `identity` | `10/13` | `0.132` | `0.265` | `0.349` | `0.000/0.474` | `0.000/0.383` | `0.508` | `3.08` |
| `joint` | `global_atom` | `11/13` | `0.116` | `0.155` | `0.254` | `0.433/0.474` | `0.266/0.383` | `0.461` | `3.08` |
| `joint` | `history_bin_atom` | `11/13` | `0.116` | `0.154` | `0.255` | `0.454/0.474` | `0.304/0.383` | `0.461` | `3.08` |

## Selected Atom Channels

- `anchor` selected: `['factor:aaa_oas', 'factor:bbb_oas']`
- `anchor` global p: `{'factor:aaa_oas': 0.43533533811569214, 'factor:bbb_oas': 0.26549482345581055}`
- `anchor` history-bin mean p: `{'factor:aaa_oas': 0.4539157450199127, 'factor:bbb_oas': 0.3046981692314148}`
- `joint` selected: `['factor:aaa_oas', 'factor:bbb_oas']`
- `joint` global p: `{'factor:aaa_oas': 0.43533533811569214, 'factor:bbb_oas': 0.26549482345581055}`
- `joint` history-bin mean p: `{'factor:aaa_oas': 0.4539157450199127, 'factor:bbb_oas': 0.3046981692314148}`

## Decision

If empirical atom gates pass BBB while preserving correlation, implement a learned hurdle/sticky gate. If they do not, the nonzero continuous path distribution must be repaired before adding a gate to the production model.

# 720a Sticky-Zero Residual Sweep

## Variant Summary

| scope | variant | pass | mean KS | AAA KS | BBB KS | AAA zero gen/gt | BBB zero gen/gt | factor corr abs ratio | conditional |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor` | `identity` | `11/13` | `0.107` | `0.286` | `0.292` | `0.000/0.474` | `0.000/0.383` | `0.616` | `4.52` |
| `anchor` | `q0.1` | `12/13` | `0.094` | `0.181` | `0.225` | `0.745/0.474` | `0.638/0.383` | `0.595` | `4.52` |
| `anchor` | `q0.5` | `12/13` | `0.094` | `0.181` | `0.225` | `0.745/0.474` | `0.638/0.383` | `0.595` | `4.52` |
| `anchor` | `q0.9` | `11/13` | `0.114` | `0.286` | `0.383` | `0.964/0.474` | `0.945/0.383` | `0.524` | `4.52` |
| `joint` | `identity` | `11/13` | `0.101` | `0.239` | `0.260` | `0.000/0.474` | `0.000/0.383` | `0.357` | `4.30` |
| `joint` | `q0.1` | `12/13` | `0.092` | `0.158` | `0.221` | `0.753/0.474` | `0.663/0.383` | `0.346` | `4.30` |
| `joint` | `q0.5` | `12/13` | `0.092` | `0.158` | `0.221` | `0.753/0.474` | `0.663/0.383` | `0.346` | `4.30` |
| `joint` | `q0.9` | `11/13` | `0.113` | `0.278` | `0.383` | `0.966/0.474` | `0.954/0.383` | `0.312` | `4.30` |

## Decision

If stronger train-derived thresholds move BBB below the KS gate without destroying correlation amplitude, sticky readout is a viable data-coordinate repair. If zero rates remain below GT or correlation collapses, the next model should use an explicit mixed discrete-continuous innovation variable rather than more threshold tuning.

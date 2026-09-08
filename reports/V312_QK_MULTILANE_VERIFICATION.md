# V3.1.2 QK Multi-Lane Verification

## Scope

- `QK_LANES=1/2/4/8` elaboration and behavior.
- Frozen score order across every lane count.
- Whole-tile causal skip with dense masked score emission.
- Inactive tail lanes when fewer column tiles than physical lanes exist.
- Random Q/K input gaps and random score/context backpressure.
- Direct QK and complete QK-to-FlashAttention-consumer integration.

## Functional Results

| Check | Result |
|---|---:|
| Direct Icarus four-lane equivalence | PASS |
| Full consumer Icarus, lanes 1/2/4/8 | PASS |
| Direct Vivado 2018.3 XSIM equivalence | PASS |
| Full consumer XSIM, lanes 1/2/4/8 | PASS |
| Score count and `score_last` | PASS |
| Request/payload stability under backpressure | PASS |
| Causal computed/skipped/masked tile counters | PASS |

The direct equivalence test checks 2048 emitted scores across two heads for
both noncausal and causal runs. All four configurations are bit-exact and emit
the same frozen tile-major coordinate order.

## Cycle Scan

| Lanes | Noncausal cycles | Speedup vs 1 | Causal cycles | Speedup vs 1 |
|---:|---:|---:|---:|---:|
| 1 | 23986 | 1.000x | 14752 | 1.000x |
| 2 | 16162 | 1.484x | 10865 | 1.358x |
| 4 | 12293 | 1.951x | 8928 | 1.652x |
| 8 | 10577 | 2.268x | 7993 | 1.846x |

## Vivado 2018.3 OOC Scan

Target: `xczu15eg-ffvb1156-2-i`, synthesized at 150 MHz with real Xilinx
Floating Point Operator checkpoints linked into the top-level netlist.

| Lanes | LUT | FF | BRAM | DSP | WNS (ns) |
|---:|---:|---:|---:|---:|---:|
| 1 | 7521 | 16440 | 0 | 66 | +4.196 |
| 2 | 14974 | 32766 | 0 | 132 | +3.644 |
| 4 | 29768 | 65418 | 0 | 264 | +3.105 |
| 8 | 59480 | 130720 | 0 | 528 | +2.231 |

These are post-synthesis OOC values, not full-system post-route sign-off.
All floating-point IP instances are included; the hierarchical reports show
each generated QK tile rather than black boxes.

## Decision

`QK_LANES=4` is the recommended integration candidate. It nearly doubles QK
throughput versus lane1 while using 8.72% LUT, 9.58% FF, and 7.48% DSP in the
QK OOC design. Lane8 remains a valid high-throughput profile, but doubles the
lane4 compute resources for only a further 1.162x noncausal speedup.

The full-system default remains unchanged until complete-system synthesis,
implementation, numerical regression, and board measurement are repeated.

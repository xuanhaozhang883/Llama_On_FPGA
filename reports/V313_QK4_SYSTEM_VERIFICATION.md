# V3.1.3 QK4 FlashAttention Verification Evidence

Date: 2026-08-03

## Configuration

- Device: XCZU15EG-FFVB1156-2-I
- Clock target: 150 MHz
- QK_LANES: 4 (default; legal values 1/2/4/8)
- V_LANES: 8
- GQA: 32 Q heads / 8 KV heads, sequence 128, head dimension 128
- Streaming path: QK tile -> online m/l update -> immediate P tile -> P tile x V tile -> O update

## Functional Results

- QK lane1/2/4/8 score equivalence and ordering: PASS, 2048 scores
- Random input gaps and output backpressure: PASS
- Whole-tile causal skip and dense masked output protocol: PASS
- QK-to-full-consumer integration for lane1/2/4/8: PASS
- Legacy FlashAttention consumer regression: 9/9 PASS
- Full-GQA numerical model: 524,288 elements, combined failures 0

## Vivado 2018.3 Results

- QK OOC synthesis lane1/2/4/8 at 150 MHz: PASS
- Full QK4/V8/8-group FlashAttention core synthesis: PASS
- Full-core post-synthesis WNS: +1.435 ns
- Full-core resources: 70,340 LUT; 126,236 registers; 97 BRAM tiles; 400 DSP
- Board-owned RTL integration synthesis: PASS with the PS/DDR Block Design black-boxed

The full-core resource report is OOC. Its 587 top-level I/O ports are not board
pins and therefore its IOB over-utilization is expected and irrelevant. All 585
methodology warnings are missing external I/O delays for this OOC boundary.

## Remaining Board Sign-off

The checked-in Block Design requires `xilinx.com:ip:zynq_ultra_ps_e:3.3`.
Vivado 2018.3 provides PS IP v3.2, so it cannot safely regenerate that vendor
design. A forced downgrade was rejected because the generated wrapper port
contract differs. Matching BIT/XSA generation, post-route timing/DRC, Vitis ELF,
and physical-board warm-up/10-run measurements remain for the teammate using a
Vivado installation that provides PS IP v3.3. Existing BIT/XSA/ELF files do not
match the v3.1.3 RTL and must not be used as v3.1.3 sign-off evidence.

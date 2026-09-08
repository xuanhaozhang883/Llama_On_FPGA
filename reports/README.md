# Verification Evidence

This directory contains the concise verification evidence retained for the v3.1.3 submission.

- `V313_QK4_SYSTEM_VERIFICATION.md`: QK4 system integration verification.
- `V312_QK_MULTILANE_VERIFICATION.md`: QK lane 1/2/4/8 equivalence summary inherited by v3.1.3.
- `V312_QK_MULTILANE_XSIM_PASS.txt`: XSIM pass record.
- `v312_qk_multilane_cycle_scan.csv`: lane-cycle comparison.
- `v312_qk_multilane_ooc_scan.csv`: lane OOC result comparison.
- `v312_qk_multilane_ooc/*/summary.txt`: per-lane timing/resource summaries.
- `v31_flash_numerical_model.json`: numerical-model result.
- `v31_flash_full_gqa_numerical.json`: full-GQA numerical result.

The historical implementation and physical-board evidence is preserved in
`../archive/release_metadata/V313_QK4_FLASHATTENTION_BOARD_VALIDATION_REPORT_CN.md`.
The cleaned current RTL still requires a new Vivado 2025.2 build and board run.

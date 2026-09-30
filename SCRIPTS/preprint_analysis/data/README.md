# Stage 2 inputs

Small, non-image inputs that Stage 2 reads besides the Stage 1 outputs. They
are referenced from [`common/config.py`](../common/config.py) through
`DATA_DIR`, with paths relative to the project root. With these files, a
checkout plus the raw experiment tree under `EXPERIMENTS/` is enough to run
`run_aggregate_results.sh`.

| Path | Used for |
|---|---|
| `c2c12_dmso_09APR26/timestamps/*.csv` | Per-frame acquisition time for C2C12 chambers A, B and C (filename, date-time, minutes). Stimulus frames are placed at the scheduled minutes from these. |
| `c2c12_dmso_09APR26/*_circle_area_filtered.npz` | The C2C12 cell selection. Each is the frame-0 Cellpose label mask after a circular region and cell-area filter were applied interactively. A cell is analysed when its frame-0 position lies on a non-zero pixel. Stored as a compressed single-array `.npz` (key `mask`, `uint16`); pixel-identical to the original `.npy`. |
| `pc3_dmso_23MAR26/PC3 DMSO pulses perfusion 23MAR26 timestamps.csv` | Per-frame acquisition time for the PC-3 recording. |
| `pc3_dmso_23MAR26/PC3 bad frames light and dark.txt` | PC-3 frames (0-based) dropped by the camera (dark) or flashed (light). They are masked and interpolated over. |
| `nrk_acid_13APR26/run_A_B/`, `run_C_D/` | Closed-loop controller logs for the two NRK runs. One run drove chambers A (channel 1) and B (channel 2), the other C (channel 1) and D (channel 2). `monitoring.log` gives the frame times, the acid decisions (stimulus frames) and the setpoint events. `luminosity_log_channel<N>.json` is the per-frame feedback record (mean fluorescence, setpoint, decision) plotted in the hardware-log figure. |

The `monitoring.log` files are the controller's logs with five strings
replaced: the ONIX server address (`<ONIX_SERVER_IP>`), the controller's
install directory (`<PIPELINE_DIR>`), the lab data directory (`<DATA_DIR>`),
the Python environment path (`<PYTHON_ENV>`) and the ONIX PC's experiment
folder (`<ONIX_EXPERIMENTS_DIR>`, also URL-encoded). No line that Stage 2
parses is changed: stimulus frames, frame times and setpoint events come out
identical for all four chambers.

`run_A_B/monitoring.log` is additionally an excerpt. The full log for that run
is 14 MB, almost all of it repeated ONIX connection errors and missing-frame
retries, so it was reduced to the lines Stage 2 reads (the 722 per-frame
decision lines `NNNNN_channelN: ...` and the two `Setpoint channelN: ...`
lines), kept verbatim and in order. Every parser output is identical to the
full log. `run_C_D/monitoring.log` is the complete scrubbed log.

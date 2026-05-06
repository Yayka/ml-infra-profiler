# MoE Production Run Timestamps

Run dir: logs/mlperf_moe_prod_20260503_175634
Run tag: prod
Config: MAX_STEPS=1000 MODEL_SIZE=moe-3b PER_DEVICE_BATCH_SIZE=1 SEQ_LENGTH=2048 TOTAL_BATCH_SIZE=8 FAKE_DATA=1
Cluster: 2 nodes (172.21.0.4 / 172.21.0.5), 2x A100 80GB PCIe each, NCCL TCP

## Launch (orchestrator wall clock — first SSH dispatch)
Local (CEST):  2026-05-03T17:56:34+0200
UTC:           2026-05-03T15:56:34Z
Unix epoch:    1777823794
Unix ms:       1777823794000

## Step 1  (first training-step log line on rank 0)
Local (CEST):  2026-05-03T17:58:54+0200
UTC:           2026-05-03T15:58:54Z
Unix epoch:    1777823934
Unix ms:       1777823934000
Log line:      step=    1  loss=10.7828  step_time=12.62s  tokens/sec=1299  lr=6.00e-06

## Training complete (orchestrator wall clock)
Local (CEST):  2026-05-03T21:26:12+0200
UTC:           2026-05-03T19:26:12Z
Unix epoch:    1777836372
Unix ms:       1777836372000
Last lines:    step= 1000  loss=10.3771  step_time=12.61s  tokens/sec=1299  lr=0.00e+00|Training complete.|

## How to use in Grafana
Time picker → Absolute → From: <Step 1 UTC>  To: <Training complete UTC>
Or via URL params: ?from=<step1_unix_ms>&to=<complete_unix_ms>

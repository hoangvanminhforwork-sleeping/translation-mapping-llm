# Queue 4: selections -> tiny rankers -> scores (CPU + short GPU bursts)
$ErrorActionPreference = "Continue"
cd "D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\autonomous-research-experiment"
$py = "..\research-scale-100k\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

& $py -m src.arms --pool pool_test_beam8.jsonl --feat feat_test_beam8.parquet *> logs\arms.log
& $py -m src.ranker *> logs\ranker.log

$arms = @(
  "results\sel_A0_greedy.jsonl", "results\sel_A1_beam.jsonl", "results\sel_A3_lenratio.jsonl",
  "results\sel_A5_mbr_chrf.jsonl", "results\sel_A6_rt_str.jsonl", "results\sel_A7_rt_lat.jsonl",
  "results\sel_A7c_rt_lat_centred.jsonl", "results\sel_A8_learned_string.jsonl",
  "results\sel_A8_learned_latent.jsonl", "results\sel_A9_oracle.jsonl"
) | Where-Object { Test-Path $_ }

& $py -m src.score --hyp @arms --boot 1000 --out metrics_10k.json *> logs\score_10k.log
& $py -m src.score --hyp @arms --boot 1000 --comet --subset core --comet-batch 16 --out metrics_core.json *> logs\score_core.log
& $py -m src.tables *> logs\tables.log
"QUEUE4_DONE" | Out-File -Encoding utf8 logs\queue4.done

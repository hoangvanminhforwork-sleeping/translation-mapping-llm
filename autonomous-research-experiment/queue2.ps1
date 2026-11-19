# Queue 2: features -> COMET labels (all GPU, sequential, one job at a time)
$ErrorActionPreference = "Continue"
cd "D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\autonomous-research-experiment"
$py = "..\research-scale-100k\.venv\Scripts\python.exe"

& $py -m src.features --pool pool_test_beam8.jsonl --out feat_test_beam8.parquet *> logs\feat_test.log
& $py -m src.features --pool pool_train_beam8.jsonl --out feat_train_beam8.parquet *> logs\feat_train.log
& $py -m src.comet_labels --pool pool_test_beam8.jsonl --out comet_test_beam8.parquet --batch 16 *> logs\comet_test.log
& $py -m src.comet_labels --pool pool_train_beam8.jsonl --out comet_train_beam8.parquet --batch 16 *> logs\comet_train.log
"QUEUE2_DONE" | Out-File -Encoding utf8 logs\queue2.done

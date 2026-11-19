# Queue 3: LABELLED CONTROL - 4M-param LoRA on opus-mt-vi-en, then the same beam-8 pool.
$ErrorActionPreference = "Continue"
cd "D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\autonomous-research-experiment"
$py = "..\research-scale-100k\.venv\Scripts\python.exe"

& $py -m src.lora_control --train --epochs 1.0 --batch 24 --lr 3e-4 --max-steps 2500 *> logs\lora_train.log
& $py -m src.gen --split test --n 10000 --mode beam --k 8 --batch 8 --model Helsinki-NLP/opus-mt-vi-en --adapter models\lora_vi_en --out pool_test_lora8.jsonl *> logs\gen_test_lora8.log
& $py -m src.gen --split test --n 10000 --mode greedy --batch 24 --model Helsinki-NLP/opus-mt-vi-en --adapter models\lora_vi_en --out pool_test_lora_greedy.jsonl *> logs\gen_test_lora_greedy.log
"QUEUE3_DONE" | Out-File -Encoding utf8 logs\queue3.done

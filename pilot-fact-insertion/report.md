# Pilot fact insertion — 15 câu, 2 arm, 1 laptop

**Trạng thái:** pilot đã chạy. **Không** phải nghiên cứu 200 câu; **không** phải kết quả cuối. Mục tiêu là kiểm tra pipeline, chọn model, và tìm chỗ hỏng trước khi scale.

---

## 0. TL;DR

| Câu hỏi | Trả lời |
|---|---|
| Laptop có chạy được không? | **Có.** `qwen2.5:1.5b` qua Ollama, full GPU, ~102.9 tok/s, ~0.53 s/câu. |
| Fact insertion có hoạt động? | **Có, ở mức pilot.** Arm A: 86.7% câu đạt ≥80% keyword; 93.3% được judge là faithful. |
| Arm “task-prompt + embedding” có tốt hơn? | **Không trong pilot này.** Arm B: 80.0% fact pass, 86.7% coverage; thấp hơn Arm A. |
| Có nên scale lên 200 câu? | **Có, nhưng sửa retrieval trước.** 2/15 lỗi retrieval kéo theo fact sai hoàn toàn. |
| Có checkpoint không? | **Không.** Chỉ code + data + kết quả. |

---

## 1. Thiết kế pilot

### 1.1 Dữ liệu

- `data/pilot_sentences.jsonl`: **15 câu** tiếng Việt, mỗi câu nêu một món ăn.
- `data/fact_bank.jsonl`: **20 fact** tiếng Việt; 15 fact đúng + 5 distractor (bún riêu, bánh tráng trộn, cà phê trứng, chè ba màu, bánh tét).
- Mỗi fact có `keywords` để đo coverage.
- Fact được viết thủ công từ kiến thức ẩm thực phổ thông; **không** coi là nguồn học thuật. Trước khi publish phải có citation.

### 1.2 Hai arm

| Arm | Retrieval | Prompt |
|---|---|---|
| **A — lexical + direct** | đếm token trùng giữa câu và fact; chọn fact có tỷ lệ trùng cao nhất | “Viết lại câu, chèn thông tin, không thêm dữ kiện” |
| **B — embedding proxy + task** | vector character 3-gram TF, khoảng cách Euclid; chọn fact gần nhất | “Nhiệm vụ: giúp người nước ngoài hiểu món ăn; nói lại câu và lồng thông tin” |

> **Quan trọng:** Arm B **không** phải neural embedding. Model Ollama hiện có không trả embedding qua `/api/embeddings`. Đây là **proxy rẻ** cho ý tưởng “vector + Euclidean”, không phải bằng chứng cho phiên bản neural. Kết quả chỉ có giá trị cảnh báo, không dùng để bác bỏ ý tưởng gốc.

### 1.3 Model và môi trường

- Runtime: **Ollama 0.40.7**, model `qwen2.5:1.5b` (986 MB trên đĩa, 1.191 GB VRAM theo `Model-Ladder-Benchmark.md`).
- Máy: i5-12500H, 8 GB RAM, RTX 3050 4 GB.
- Lý do chọn: model **đã cài**, chạy full GPU, chất lượng tiếng Việt tốt hơn `gemma3:1b`/`270m` trong thử nghiệm nhanh. Đây là **stand-in** cho Qwen3.5-0.8B/LFM2.5-350M/Apertus 0.5B — ba model này **chưa** được cài/đo trong pilot.
- Nhiệt độ 0.1; `num_predict=120`; seed mặc định.

---

## 2. Kết quả

### 2.1 Retrieval (chỉ chọn fact)

| Phương pháp | Top-1 accuracy | Lỗi |
|---|---:|---|
| Lexical | **0.933** (14/15) | S03 |
| Embedding proxy (char 3-gram + Euclid) | **0.933** (14/15) | S14 |

Hai phương pháp **không sai giống nhau**. Lexical nhầm S03 (bánh mì) sang bò kho vì cùng có “bún/bánh mì”; embedding proxy nhầm S14 (hủ tiếu) sang bánh xèo vì trùng “bột gạo, miền Nam”.

**Kết luận:** retrieval là **nút thắt số 1**. Nếu chọn sai fact, generation có tốt đến đâu cũng sai.

### 2.2 Generation

| Metric | Arm A (lexical + direct) | Arm B (embedding + task) |
|---|---:|---:|
| Fact pass (≥80% keyword) | **0.867** | 0.800 |
| Mean keyword coverage | **0.911** | 0.867 |
| Judge faithful rate | **0.933** | **0.933** |
| Judge fluent (1–5) | **4.067** | 3.867 |
| One-sentence rate | 0.933 | 0.933 |
| Mean generation time | 0.529 s | 0.509 s |
| Mean tokens/s | **102.9** | 101.3 |

### 2.3 Ví dụ

**S01 — Arm A (đúng, tự nhiên):**

> Phở là món nước nổi tiếng của Việt Nam, thường được ăn vào buổi sáng, đặc biệt là với nước dùng phở bò Hà Nội được ninh từ xương bò nhiều giờ, thêm gừng nướng, quế và hồi.

**S03 — Arm A (retrieval sai, fact sai):**

> Bánh mì Việt Nam là món ăn đường phố quen thuộc với du khách, thường ăn kèm với bò kho hầm với cà rốt, sả, gừng và gia vị, hoặc bún.

Fact đúng là bánh mì Pháp/vỏ giòn; nhưng retrieval chọn bò kho. Câu văn trôi, nhưng **sai thông tin**. Đây là lỗi nguy hiểm nhất: fluent nhưng unfaithful.

**S03 — Arm B (retrieval đúng, fact đúng):**

> Bánh mì Việt Nam, món ăn đường phố quen thuộc với du khách, là sự kết hợp của bánh mì Pháp với nhân thịt, chả hoặc trứng, tạo nên một món ăn giòn và hấp dẫn.

**S14 — Arm B (retrieval sai):**

> Hủ tiếu là món nước có sợi bánh làm từ gạo, phổ biến ở miền Nam, thường ăn kèm với bánh xèo miền Nam có bột gạo pha nghệ, nhân tôm thịt và giá đỗ, ăn kèm rau sống.

Lại là fluent + sai fact. Đây là lý do phải đo **faithfulness**, không chỉ đọc thấy mượt.

---

## 3. Laptop có chịu được không?

### 3.1 Đo được

- `qwen2.5:1.5b`: **102.9 tok/s**, **0.53 s/câu** (15 câu, 2 arm, tổng khoảng 16 giây generation; cộng judge khoảng 30 giây).
- VRAM: 1.191 GB theo benchmark; fits 4 GB.
- RAM: ~1.5 GB free trong lúc chạy; tight nhưng ổn nếu đóng app nặng.
- 15 câu × 2 arm + judge: **dưới 2 phút**.

### 3.2 Chưa đo

- Qwen3.5-0.8B, LFM2.5-350M, Apertus 0.5B: chưa cài, chưa chạy. Không được nói “chạy được” cho các model này.
- 200 câu: chưa chạy. Ước lượng từ pilot: 200 × 2 arm × 0.53 s ≈ 3.5 phút generation; cộng judge/retrieval/I/O có thể 10–20 phút. RAM/VRAM không đổi.
- Fine-tune: không nằm trong pilot này.

### 3.3 Kết luận

**Inference 1.5B: thoải mái.** Nếu chỉ cần pipeline fact insertion 200 câu, laptop này thừa sức. Nếu cần model 3B+, sẽ offload CPU và chậm ~6× (theo `Model-Ladder-Benchmark.md`).

---

## 4. Vì sao Arm B chưa thắng

Ba lý do, theo thứ tự quan trọng:

1. **Retrieval proxy yếu.** Character 3-gram không hiểu “hủ tiếu” khác “bánh xèo”; nó chỉ thấy trùng ký tự. Neural embedding (bge-m3, multilingual-e5) sẽ khác.
2. **Prompt task không tự tạo fact.** Cùng một fact đúng, prompt task có thể làm câu dài hơn, nhưng không sửa được fact sai. Trong pilot, Arm B có 3/15 retrieval sai (S03? no, S14) → fact pass thấp.
3. **Judge cùng model.** Arm B không được judge ưu ái; nhưng judge cũng là qwen2.5:1.5b, nên có self-preference. Cần judge độc lập (Qwen3.5-0.8B hoặc người) trước khi kết luận.

**Không** kết luận “task-prompt embedding vô dụng”. Chỉ kết luận: **proxy này không đủ**; muốn test ý tưởng thật phải dùng embedding neural và map tuyến tính/MLP.

---

## 5. Nếu scale lên 200 câu

### 5.1 Sửa gì trước

1. **Retrieval:** thay lexical/char n-gram bằng embedding đa ngữ (bge-m3, multilingual-e5-small) hoặc BM25 + fuzzy match; đo top-1 trên 50 câu.
2. **Fact bank:** 200 fact, có `source` và `citation`; tách fact đúng/sai.
3. **Evaluation:** thêm judge độc lập; human spot-check 20%; đo unsupported-claim rate, không chỉ keyword coverage.
4. **Prompt:** few-shot 2–3 ví dụ; yêu cầu giữ nguyên tên món và không thêm fact ngoài.
5. **Logging:** lưu prompt, output, retrieval score, latency, model version.

### 5.2 Ước lượng chi phí

- 200 câu × 2 arm × 0.53 s ≈ 3.5 phút generation.
- Judge 400 lần × ~1 s ≈ 7 phút.
- Retrieval + I/O + retry: 10–20 phút.
- RAM/VRAM: không đổi.

### 5.3 Tiêu chí dừng

- Fact pass < 0.80 trên 200 câu → không scale; sửa retrieval/prompt.
- Unsupported claim > 10% → dừng, thêm constraint.
- Retrieval top-1 < 0.90 → dừng, đổi retriever.

---

## 6. Giới hạn

- 15 câu, 20 fact: quá nhỏ để nói về tổng thể.
- Fact do người viết, chưa có citation.
- Judge cùng model; có thể thiên vị.
- Arm B là proxy, không phải neural embedding.
- Chưa đo các model trong danh sách chính thức.
- Chưa có human evaluation.
- Keyword coverage không bắt được fact sai kiểu “đúng từ nhưng sai quan hệ”.

---

## 7. File

```text
pilot-fact-insertion/
  data/pilot_sentences.jsonl    # 15 câu
  data/fact_bank.jsonl          # 20 fact
  run_pilot.py                  # pipeline + judge
  results/pilot_results.json    # từng câu, từng arm
  results/pilot_summary.json    # bảng tổng hợp
  report.md                     # file này
```

Chạy lại:

```powershell
# cần Ollama đang chạy và model qwen2.5:1.5b
cd D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\pilot-fact-insertion
python run_pilot.py
```

> Kết quả là **pilot**, không phải kết quả công bố. Dùng để quyết định có nên scale, không dùng để claim “hệ thống hoạt động”.

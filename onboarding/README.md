# Onboarding — Culinary CSI

Trang onboarding nội bộ cho dự án **Culinary CSI Translation Through Dynamic Fact Insertion**. Mục tiêu: một teammate biết ML cơ bản nhưng mới vào dự án có thể đọc trong khoảng 25 phút và nắm được:

1. Paper 1 — ICOCO: representation ẩm thực 304-D, K-means “battle”, baseline FoodBERT, các claim và con số.
2. Paper 2 — translation through dynamic fact insertion: ánh xạ representation giữa hai model đơn ngữ, hai experiment nhỏ đã xong, bằng chứng hiện có và cái gì **chưa** được chứng minh.
3. Phần **bất biến** (abstraction) so với phần **biến** (implementation): model, lớp trung gian, pooling, layer, độ rộng kênh.
4. Cách tái lập: môi trường, dữ liệu, lệnh chạy, nơi lưu kết quả.
5. Lộ trình đọc và glossary.

## Mở trang

Cách 1 — mở trực tiếp bằng trình duyệt mặc định:

```powershell
Start-Process "D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\onboarding\index.html"
```

Cách 2 — mở bằng Microsoft Edge:

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "file:///D:/HANU_FIT/NCKH/Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion/onboarding/index.html"
```

Không cần internet. Trang tự chứa CSS + JS + SVG, không webfont, không CDN, không thư viện ngoài.

## File trong thư mục

| File | Vai trò |
|---|---|
| `index.html` | Toàn bộ website: HTML + CSS + JS + SVG inline. Tự chứa. |
| `onboarding.pdf` | Bản in PDF do Microsoft Edge headless tạo. Có thể lệch vài chỗ so với bản web vì print CSS. |
| `README.md` | File này. |

Không tách `style.css` / `app.js` để tránh phụ thuộc đường dẫn tương đối khi copy file hoặc mở qua `file://`.

## Cấu trúc website

| Mục | Nội dung |
|---|---|
| Hero | Định vị dự án, 3 tầng: representation → ánh xạ → fact insertion. |
| 00 · 60 giây | Bài toán, ý tưởng, đã có gì, chưa có gì; quy ước trạng thái. |
| 01 · Paper 1 (ICOCO) | 50.510 recipe; 304-D = 248 geo + 28 entity + 28 odor; K-means K ∈ {6, 31, 207}, 5 seeds, Hungarian; PA / Soft-MacroF1; FoodBERT; probe food/non-food; audit nội bộ. |
| 02 · Paper 2 | Hai experiment: mapping + probing (2026-10-21) và vector→token (2026-10-22); bảng BLEU; kết quả âm tính của inversion; fact insertion là đề xuất. |
| 03 · Kiến trúc | Sơ đồ EN→VI và VI→EN; bất biến vs biến; điều kiện model phía Việt phải sinh được token Việt; nút thắt kênh 768-D. Có điều khiển tương tác đổi cặp model / lớp trung gian. |
| 04 · Bằng chứng | 17 dòng, lọc theo trạng thái: đã chứng minh / có điều kiện / đề xuất / chưa chứng minh. |
| 05 · Tái lập | Môi trường, dữ liệu, lệnh PowerShell, nơi lưu kết quả, checkpoint kiểm tra. |
| 06 · Lộ trình đọc | 8 bước có checkbox lưu tiến độ bằng `localStorage`; gợi ý đường 15 phút. |
| 07 · Glossary | 24 thuật ngữ. |

## Nguồn đã đọc và dùng

Tất cả nằm trong `D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion` trừ ICOCO.

| Nguồn | Dùng cho |
|---|---|
| `paperOverview.md` | Kiến trúc mong muốn, câu hỏi nghiên cứu, các quyết định còn mở. |
| `PROJECT_WORKFLOW.md` | Trạng thái thực tế, hai experiment, bảng BLEU, ranh giới claim, cổng quyết định. |
| `present-Saturday-24th Oct/content.md` | Tóm tắt 10 phút, probing, ý tưởng fact insertion bằng perplexity, ghi chú “8M cặp câu”. |
| `small-experiment/results/summary.md` | Mapping EN→VI, R@1/R@10, probe food/non-food, alpha follow-up. |
| `small-experiment/results/mapping_metrics.csv` | R@1 74,1%; R@10 92,2%; identity 0,1%. |
| `small-experiment/results/probe_food.json` | Accuracy 0,99265; F1 0,992647; AUC 0,9997316. |
| `small-experiment-02/results/summary.md` | Bảng BLEU/chrF, ratios, negative result inversion, threats to validity. |
| `small-experiment-02/results/nn_metrics.csv` | Số BLEU gốc của 5 nhánh. |
| `small-experiment/REPRODUCE.md` | Môi trường, Docker MySQL, lệnh experiment 01. |
| `small-experiment-02/REPRODUCE.md` | Lệnh experiment 02, checkpoint, trap NLTK. |
| `ICOCO_reproducible/final/presentation/paper_facts.md` | 50.510 recipe; 30.621 labelled; 116 tags; 3.500 ingredients; 304-D; metric canonicalization; silhouette/DB; claim FoodBERT. |
| `ICOCO_reproducible/final/huge_notebook_docs_note.md` | Thành phần 304-D; K-means K ∈ {6, 31, 207}; 5 seeds; Hungarian; metric. |
| `ICOCO_reproducible/final/paper_audit.md` | Audit nội bộ: metric nội bộ tái lập; external claim không tái lập; country SF1 0,0951/0,0539 = 1,76× so với 0,0618/0,0542 = 1,14×; Entity 26,07 và No-Entropy 29,18 < FoodBERT 30,07 tại K=6. |
| `Local_LLM_runable/Kokoro-82M-Local-Benchmark.md` | Chỉ dùng để đối chiếu cấu hình máy (i5-12500H, 8 GB RAM, RTX 3050 4 GB) khi nói về giới hạn tài nguyên. Không dùng số TTS. |

## Những con số chưa chắc chắn / cần xác minh

Các mục này được đánh dấu trên website, không bị coi là “đã chứng minh”:

1. **“8 triệu cặp câu”** — chỉ xuất hiện trong ghi chú họp `present-Saturday-24th Oct/content.md` (“Chưa full dataset: 8M cặp câu”). Chưa đối chiếu với nguồn dữ liệu gốc. Trang đánh dấu **cần xác minh**.
2. **50.510 vs 50.511 vs 50.513** — paper dùng 50.510; `MasterBattle.csv` có 50.513 dòng; các file `.npy` hiện tại có 50.511 dòng (theo `paper_audit.md` §1). Trang dùng **50.510** theo `paper_facts.md` và không che chênh lệch này. Nếu trích cho publication, cần chốt lại.
3. **Country-level SF1 “xấp xỉ 2×”** — paper báo 0,0951/0,0539 = 1,76×; audit dưới metric nhất quán báo 0,0618/0,0542 = 1,14×. Trang trình bày **cả hai**, không chọn một con số.
4. **“Mọi variant vượt FoodBERT ở PA”** — paper tuyên bố; audit nói sai tại K=6. Trang đánh dấu **cần xác minh**.
5. **“~25 phút đọc”, thời gian từng bước trong lộ trình** — ước lượng của người biên soạn, không đo.
6. **“RAM còn ~1,6 GB”** — quan sát từ phiên làm việc hiện tại, không phải số trong repo. Chỉ dùng để cảnh báo không chạy pipeline nặng.
7. **R@1 = 76,0%** — lấy từ `small-experiment/results/summary.md` §3 (ridge α=100, follow-up). Bảng chính `mapping_metrics.csv` dùng α=1000 và ghi 74,1%. Trang nói rõ hai cấu hình.
8. **Các con số BLEU tuyệt đối** — readout là retrieval trên gallery 10k, không phải generation; không so sánh được với BLEU trong literature MT. Trang ghi rõ.
9. **ANN +0,6% / +6,5% BLEU** — lấy từ `small-experiment-02/results/summary.md` §2; summary cũng nói chênh lệch 0,02 BLEU nằm trong nhiễu và **chưa có significance test**. Trang không claim ANN thắng.

## Đã kiểm tra

- `index.html`: **107.465 bytes**, 1.476 dòng, UTF-8, không có `http://`, `https://`, `@import` hay webfont. Các `url(...)` còn lại là SVG gradient nội bộ (`url(#plateGrad)`).
- 56 `id`, không trùng; 15 anchor `href="#..."`, không thiếu đích.
- Render bằng Microsoft Edge headless:
  - PDF: `onboarding.pdf` — **861.431 bytes** (lần render đầu; file có thể được ghi đè khi render lại).
  - Screenshot desktop 1440 px và mobile 520 px đã được kiểm tra thủ công.
- Tương tác đã kiểm tra: đổi `Cặp model` sang Decoder-only + `Lớp trung gian` sang k-token prefix cập nhật đúng sơ đồ, nhãn và chip trạng thái; bảng bằng chứng lọc được; checkbox lộ trình lưu `localStorage`.
- Không sửa file nguồn nào. Chỉ tạo thư mục `onboarding/` và các file trong đó.

### Lệnh render PDF đã dùng

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" `
  --headless=new --disable-gpu --no-first-run --no-default-browser-check `
  --user-data-dir="D:\HANU_FIT\BigTasks\agent-work\m2a\edge-profile" `
  --no-pdf-header-footer `
  --print-to-pdf="D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\onboarding\onboarding.pdf" `
  "file:///D:/HANU_FIT/NCKH/Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion/onboarding/index.html"
```

### Lệnh render screenshot đã dùng

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" `
  --headless=new --disable-gpu --no-first-run --no-default-browser-check `
  --user-data-dir="D:\HANU_FIT\BigTasks\agent-work\m2a\edge-profile-shot" `
  --hide-scrollbars --window-size=1440,1100 `
  --screenshot="D:\HANU_FIT\BigTasks\agent-work\m2a\onboarding-top.png" `
  "file:///D:/HANU_FIT/NCKH/Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion/onboarding/index.html"
```

## Ghi chú biên tập

- Trang tách rõ **Paper 1 đã công bố / đã tái lập** và **Paper 2 là hướng đang làm, chưa công bố**.
- “Đã chứng minh” chỉ có nghĩa trong phạm vi thí nghiệm tương ứng; không mở rộng sang generation, culinary domain hay văn hoá.
- Các kết quả âm tính (embedding inversion; ANN không vượt ridge) được giữ lại thay vì ẩn đi, vì chúng quyết định bước tiếp theo.
- Nếu cập nhật số liệu, sửa trực tiếp trong `index.html` và render lại PDF. Không có build step.

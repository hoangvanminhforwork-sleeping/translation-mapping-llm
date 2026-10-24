# PROJECT WORKFLOW — trạng thái thực tế (2026-10-24)

Tài liệu này mô tả **dự án đang ở đâu**, không phải dự án *muốn* đi đâu.
Mọi con số dưới đây lấy từ `results/` đã commit, không phải ước lượng.

Nguồn gốc yêu cầu: `paperOverview.md`. Bằng chứng chi tiết:
`small-experiment/results/`, `small-experiment-02/results/summary.md`.

## 0. Chú thích (đọc trước khi xem hình)

| ký hiệu | nghĩa |
|---|---|
| nét liền, xanh | **đã chạy, đã đo**, có file kết quả trong `results/` |
| nét đứt, vàng | **đã thiết kế, chưa chạy** (spec có trong `HANDOFF.md`) |
| nét chấm, xám | **chưa bắt đầu** (chỉ là đề xuất trong `paperOverview.md`) |
| `[số]` | thứ tự stage trong pipeline, chạy được độc lập |

Nguyên tắc bất biến của cả hai experiment: seed 42, cặp câu đóng băng ra
parquet + SHA-256, mọi stage chạy lại được (idempotent), vector không bao giờ
được vẽ lại giữa chừng.

## 1. Ý tưởng của paper (kiến trúc *mong muốn*)

```mermaid
flowchart LR
    A["Câu nguồn<br/>EN: 'the soup is sour'"] --> B["Model nguồn<br/>frozen"]
    B --> C["Hidden state<br/>h_en ∈ R^768"]
    C --> D["ANN ánh xạ f<br/>THAM SỐ DUY NHẤT ĐƯỢC TRAIN"]
    D --> E["Vector đã map<br/>f(h_en) ∈ R^768"]
    E --> F["Model đích<br/>frozen"]
    F --> G["Câu dịch<br/>'canh chua'"]
    G --> H["Chèn fact ẩm thực<br/>(RAG, Stage 4)"]
    H --> I["Câu dịch có ngữ cảnh văn hoá"]

    classDef train fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    classDef todo fill:#eeeeee,stroke:#999,stroke-dasharray:3 3
    class D train
    class H,I todo
```

Điểm cốt lõi: **không train lại LLM**, chỉ train mạng ánh xạ. Vì vậy câu hỏi
nghiên cứu không phải "dịch có tốt không" mà là **"một vector 768 chiều có đủ
làm kênh truyền ngôn ngữ hay không"**.

## 2. Thực tế đã làm — hai experiment nhỏ

```mermaid
flowchart TB
    subgraph E1["small-experiment — Stage 1: mapping + probing (ĐÃ XONG)"]
        direction LR
        A1["12k cặp EN–VI<br/>general domain"] --> B1["2 bi-encoder frozen<br/>bge-en + bkai-vi<br/>768/768"]
        B1 --> C1["3 pooling<br/>mean / cls / last"]
        C1 --> D1["ridge vs MLP<br/>vs identity"]
        D1 --> E1a["Đo: MSE, cosine, R@1"]
        F1["10k title RecipeNLG<br/>+ 10k câu thường"] --> G1["logistic regression<br/>food / not-food"]
    end

    subgraph E2["small-experiment-02 — Stage 2: vector có đọc ra chữ không (ĐÃ XONG)"]
        direction LR
        A2["12k cặp kế thừa<br/>SHA-256 đóng băng"] --> B2["encode lại fp32<br/>6 mảng .npy"]
        B2 --> C2["refit ridge alpha=100<br/>+ MLP 768-2048-768"]
        C2 --> D2["readout A: retrieval<br/>gallery 10k câu train"]
        C2 --> E2a["readout B: embedding<br/>inversion (THẤT BẠI)"]
        D2 --> F2["BLEU/chrF/TER/<br/>NIST/ROUGE-L/EM"]
    end

    subgraph E3["Chưa làm"]
        direction LR
        A3["decoder sinh token<br/>(R2, spec xong)"]:::dashed
        B3["neural metrics<br/>COMET/BERTScore"]:::dashed
        C3["corpus ẩm thực<br/>+ RAG chèn fact"]:::todo
        D3["survey độ quen<br/>món ăn"]:::todo
    end

    E1 --> E2
    E2 -.-> E3

    classDef dashed fill:#fff8e1,stroke:#f9a825,stroke-dasharray:5 5
    classDef todo fill:#eeeeee,stroke:#999,stroke-dasharray:3 3
```

> Điểm phải nhớ: **paper nói "hidden state của decoder"**, nhưng hai experiment
> chỉ đo trên **sentence embedding của encoder**. Hai checkpoint có sẵn trên ổ
> đĩa là encoder-only (không `lm_head`), nên mũi tên "model đích sinh chữ" ở
> §1 **chưa từng được test**.

## 3. Pipeline chi tiết — `small-experiment/` (Stage 1)

```mermaid
flowchart LR
    S0["data/data/train-00008-of-00014.parquet<br/>594.836 dòng"] --> S1
    R0["data/RecipeNLG.csv<br/>2.231.142 dòng"] --> S1
    S1["[1] sample<br/>seed 42, lọc ≤64 từ<br/>→ 267.475 cặp sạch"] --> A1["artifacts/sample_pairs.parquet<br/>12k = 10k train + 2k test"]
    A1 --> S2["[2] encode<br/>fp16, max 128 token"]
    S2 --> DB[("MySQL 9.4 · Docker · :3307<br/>pairs + embeddings<br/>72.000 vector VECTOR(768)")]
    DB --> S3["[3] verify<br/>số dòng, dim, sai số 1e-6"]
    DB --> S4["[4] mapping<br/>ridge / MLP / identity"]
    DB --> S5["[5] probe<br/>logistic regression"]
    S4 --> R4["results/mapping_metrics.csv<br/>results/retrieval_at1.png"]
    S5 --> R5["results/probe_food.json"]
```

**Kết quả đo được** (closed-set, 2.000 test):

| phương pháp | R@1 | R@10 | ghi chú |
|---|---|---|---|
| identity (không map) | 0,1% | 0,85% | sàn — vector nằm sai không gian |
| **ridge** | **74,1%** | **92,2%** | thắng ở mọi pooling |
| MLP (ANN) | 66,0% | 90,5% | thêm phi tuyến **không đáng** với 9k cặp |

Probe food/not-food: **accuracy 99,3%, AUC 0,9997** → thông tin ẩm thực mã hoá
tuyến tính trong không gian EN. *Cảnh báo confound*: positive là cụm danh từ
ngắn, negative là câu đầy đủ → probe có thể thắng nhờ **style**, chưa chắc nhờ
ngữ nghĩa món ăn.

## 4. Pipeline chi tiết — `small-experiment-02/` (Stage 2)

```mermaid
flowchart TB
    P["artifacts/sample_pairs.parquet<br/>(từ exp 01, không vẽ lại)"] --> C["[cache] 129 s<br/>6 mảng [12000, 768]"]
    C --> M["[maps] 25 s<br/>12 file map"]
    M --> N["[nn] 492 s<br/>5 nhánh × 6 combo"]
    N --> R["results/nn_metrics.csv<br/>30 dòng × 8 metric"]

    C --> I["[invert] 1631 s<br/>64 item × 100 step"]
    I --> RI["results/hyp_*.parquet<br/>KẾT QUẢ ÂM TÍNH"]

    R --> J{"Có tách được<br/>các nhánh?"}
    J -->|có| K["retrieval là readout hợp lệ"]
    J -->|không| L["dừng, đừng scale lên"]

    style I fill:#ffcdd2,stroke:#c62828
    style RI fill:#ffcdd2,stroke:#c62828
```

**5 nhánh so sánh** (bắt buộc giữ `shuffled` trong mọi readout sau này):

`roundtrip` (trần trên, vector oracle) · `identity` · `ridge` · `ann` ·
`shuffled` (vector lấy từ cặp câu khác — đối chứng "vector hợp lý nhưng sai").

BLEU, mean pooling:

| nhánh | en→vi | vi→en |
|---|---|---|
| roundtrip (trần) | 5,10 | 2,54 |
| **ridge** | **3,27** | **1,35** |
| ann | 3,29 | 1,44 |
| identity | 0,09 | 0,04 |
| shuffled | 0,24 | 0,13 |

Ba kết luận đã có bằng chứng:

1. **Vector sau map vẫn mang nội dung câu**: 3,27 BLEU = **13,7×** shuffled.
2. **Mất chữ, không mất nghĩa**: giữ 87–92% chrF của trần nhưng chỉ 53–64% BLEU.
3. **ANN chưa xứng đáng**: ridge ≥ ann trên mọi trục đo.

Và một lỗi khoa học đã sửa: không gian EN dị hướng mạnh (‖μ‖ = 10,8) nên hai
vector **ngẫu nhiên** đã có cosine 0,52. Con số `vi2en` = 0,80 của exp 01 là
**ảo giác**; sau khi trừ mean còn 0,52 → hai chiều đối xứng, không có bất đối
xứng như tưởng.

## 5. Cổng quyết định — bước tiếp theo

```mermaid
flowchart TD
    Q["Kênh 768-d đã chạm trần:<br/>oracle chỉ 5,10 BLEU qua retrieval"] --> D1{"Muốn tăng số<br/>thì tăng cái gì?"}
    D1 -->|"map to hơn"| X["Bác bỏ:<br/>ridge đã ≥ ANN"]
    D1 -->|"kênh rộng hơn"| Y["k-token prefix<br/>hoặc decoder hidden state"]
    D1 -->|"đo cho chắc"| Z["R2: decoder 4 lớp,<br/>cross-attention vào 1 vector<br/>~20 phút/chiều"]
    Y --> N1["mở paperOverview §3.3"]
    Z --> N2["spec đầy đủ: HANDOFF.md §6"]
    N1 --> N3["chưa chạy"]
    N2 --> N3

    style X fill:#ffcdd2,stroke:#c62828
    style N3 fill:#eeeeee,stroke:#999,stroke-dasharray:3 3
```

Logic của nhánh `R2` đáng nhớ vì nó *quyết định được*: nếu `ridge ≈ novec`
(vector không đóng góp gì) → map vô dụng; nếu `oracle ≫ ridge` → map làm mất
thông tin; nếu **mọi nhánh bằng nhau** → nút thắt là **kênh**, không phải map.

## 6. Bản đồ file — đọc gì trước

| muốn biết | đọc |
|---|---|
| mục tiêu, giới hạn, câu hỏi nghiên cứu | `paperOverview.md` |
| thí nghiệm 1 làm gì, giả định nào | `small-experiment/agent-work/ASSUMPTIONS.md` |
| thí nghiệm 1 chạy lại thế nào | `small-experiment/REPRODUCE.md` |
| **bằng chứng mới nhất** | `small-experiment-02/results/summary.md` |
| trạng thái + việc kế tiếp | `small-experiment-02/agent-work/HANDOFF.md` |
| bài học kỹ thuật (bẫy đã sập) | `small-experiment-02/agent-work/MISSION_NOTE.md` |
| lệnh chạy lại exp 02 | `small-experiment-02/REPRODUCE.md` |

## 7. Ranh giới của những gì đã chứng minh

**Được phép nói:**
- Một bi-encoder EN có thể được **map tuyến tính** sang không gian bi-encoder VI
  với R@1 = 74% (so với 0,1% khi không map).
- Vector sau map **còn đọc được nội dung câu**, không chỉ chủ đề.
- Thông tin "đây là món ăn" **tuyến tính** trong embedding EN.

**Không được nói:**
- "Hệ dịch hoạt động" — chưa có decoder nào sinh câu.
- "ANN là cần thiết" — ridge đã đủ và tốt hơn.
- "Dịch ẩm thực" — corpus là general domain (pháp lý/hằng ngày).
- "Chuyển được văn hoá" — chưa có thí nghiệm nào về văn hoá.
- So sánh BLEU ở đây với BLEU trong literature MT — readout là **retrieval**,
  trần do gallery 10k quyết định.

## 8. Phần chưa chạm tới

| hạng mục | trạng thái | phụ thuộc |
|---|---|---|
| decoder sinh token (R2) | thiết kế xong, chưa chạy | user đã dừng |
| neural metrics (COMET, BERTScore…) | chưa tải gì | cần đo feasibility từng cái |
| significance test (bootstrap) | chưa | cần trước khi công bố chênh lệch nhỏ |
| corpus ẩm thực thật | chưa | thay `train-00008.parquet` |
| RAG chèn fact (Dynamic Fact Insertion) | chưa | cần corpus + retrieval |
| survey độ quen thuộc món ăn | chưa | hướng nghiên cứu **tách rời** |

8. Thí nghiệm tao đề xuất: Tách decoder ra để kiểm tra
Thay vì lập tức train translation lâu hơn, tao sẽ làm một bộ diagnostic gồm 4 thí nghiệm.

Giữ nguyên ViT5 encoder, W và Flan-T5-base.
Experiment	Input vào decoder	Target
A	Native Flan states	Tiếng Việt
B	Mapped ViT5 states	Tiếng Việt
C	Native Flan states	Tiếng Anh
D	Mapped ViT5 states	Tiếng Anh

Trong đó:
- A, B kiểm tra khả năng reconstruction.
- C, D kiểm tra translation.
Kết quả sẽ giúp phân biệt các tình huống:
Cách đọc kết quả
A tốt, B kém
Mapping interface

Decoder đọc được native nhưng chưa đọc được mapped representation.

A tốt, B tốt, D kém
Translation task

Decoder đọc được mapped vectors nhưng chưa học tốt cách dịch.

A kém, B kém
Decoder / training

Cần kiểm tra decoder, training objective và khả năng reconstruction.

B tốt, D tốt
Promising

Mapping và decoder đều hoạt động tương đối tốt.

Một lưu ý: reconstruction và translation cần adapter/objective được thiết kế phù hợp; không nên lấy một checkpoint đã train cho translation rồi đòi nó tự nhiên biết copy tiếng Việt.

---

Bro, khoan. Tao nghĩ mày vừa chạm đúng vào một vấn đề mà phần phân tích trước của tao chưa đánh giá đủ nghiêm túc.

Và tao muốn sửa lại một nhận định:

**Chưa có cơ sở để nói decoder là nút thắt chính. Encoder + mapping hoàn toàn có thể đang là nút thắt.**

Thậm chí, với những số liệu hiện tại, tao nghĩ cần kiểm tra chất lượng representation trước khi đổ thêm tài nguyên vào decoder.

Có ba vấn đề cần tách biệt.

# **1. Cosine 0.751 có thực sự tốt không?**

Mày nói đúng: decoder không phải một hệ thống có khả năng chịu sai lệch vector vô hạn.

Nhưng cần hiểu cosine 0.751 đang đo cái gì.

Nó đo:

\\[ \cos(XW+b,Y)=0.751 \\]

Tức là góc giữa hai vector tương ứng sau mapping và vector target trung bình theo sentence-level evaluation.

Góc tương ứng:

\\[ \theta=\arccos(0.751)\approx 41.3^\circ \\]

**Hình dung độ lệch hướng**

Minh họa hai vector có cosine 0.751, không phải visualization thực tế của embedding 768 chiều.

41 độ không phải sai lệch nhỏ.

Nhưng cũng không thể kết luận decoder không sử dụng được chỉ từ góc này.

Vì transformer không chỉ đọc cosine similarity.

Nó có các projection matrices \\(W_K,W_V\\), rồi nhiều lớp attention và feed-forward để xử lý representation.

Nó có thể chịu được một số dạng biến đổi mà cosine không phản ánh đầy đủ.

**Điều quan trọng hơn là sai lệch đó có làm thay đổi các phép tính bên trong decoder hay không.**

Ví dụ, nếu mapped vector có sai lệch lớn nhưng vẫn giữ được các đặc trưng mà decoder cần, nó có thể hoạt động.

Ngược lại, cosine 0.95 vẫn có thể thất bại nếu sai lệch tập trung đúng vào những chiều quan trọng đối với decoder.

# **2. Retrieval\@1 = 48.75% có đáng lo không?**

Có. Và tao nghĩ đây là chỉ số cần đào sâu nhất.

Nhưng phải hiểu chính xác nó đang đo gì.

Giả sử có 2.000 câu trong test gallery.

Với mỗi vector được mapping, hệ thống tìm vector tiếng Anh gần nhất.

Nếu tìm đúng câu tương ứng thì tính một điểm.

Kết quả 48.75% nghĩa là chỉ khoảng 975/2.000 câu tìm được đúng câu tương ứng ở vị trí đầu tiên.

**2.000 câu kiểm tra**

975 đúng

1.025 chưa đúng

Điều đó có nghĩa mapping chưa bảo toàn hoàn toàn danh tính của từng câu trong không gian đích.

Nhưng có một điểm cần lưu ý:

**Retrieval\@1 không phải tỷ lệ token được dịch đúng, cũng không phải tỷ lệ thông tin được giữ lại.**

Một câu có thể tìm nhầm câu khác nhưng vẫn giữ phần lớn semantic content.

Ví dụ:

- Câu A: Tôi thích ăn phở.
- Câu B: Tôi thường ăn phở.

Hai câu có thể rất gần nhau về embedding, nhưng retrieval vẫn tính sai nếu target chính xác là A.

Vậy nên 48.75% là một tín hiệu đáng lo về khả năng phân biệt câu, nhưng chưa đủ chứng minh 51.25% còn lại bị mất thông tin.

Một câu hỏi quan trọng nữa: gallery có bao gồm những câu gần nghĩa, câu trùng nghĩa hoặc các bản dịch có nhiều cách diễn đạt khác nhau không?

Nếu có, retrieval\@1 có thể đánh giá khắt khe hơn thực tế.

# **3. Vậy decoder có cần được LoRA lại phần copy không?**

Đây mới là chỗ tao muốn phân tích kỹ nhất.

Mày nói:

> Khi làm LoRA copy, model đã đạt 99.8%. Tại sao lại cần LoRA phần nhại lại thêm?

Tao đồng ý với mày rằng nếu mục đích chỉ là chứng minh model có thể copy input thông thường thì không có lý do gì phải train lại một lần nữa.

Nhưng có một sự khác biệt:

**Copy experiment**

Vietnamese text

Flan encoder

Flan decoder + copy LoRA

Vietnamese text

**Mapped translation experiment**

Vietnamese text

ViT5 encoder

Linear mapping W

Flan decoder + translation LoRA

English translation

Hai pipeline có thể đưa vào decoder những hidden states rất khác nhau.

Và đây là chỗ quan trọng:

Trong T5, encoder và decoder được pretrained cùng nhau. Decoder đã học cách sử dụng output của chính encoder đó.

Nó không mặc định biết cách sử dụng output của một encoder khác, kể cả khi hai encoder có cùng kích thước 768.

Tuy nhiên, mapping được tạo ra chính để giải quyết sự khác biệt này.

Vì vậy, ta có hai khả năng:

**Khả năng A:** Mapping chưa đủ tốt để đưa representation vào đúng vùng mà decoder sử dụng hiệu quả.

**Khả năng B:** Mapping đã đủ tốt, nhưng decoder cần điều chỉnh cách sử dụng representation đó.

Đây là hai giả thuyết cần thực nghiệm phân biệt.

Và tao không nghĩ nên tiếp tục LoRA copy trên native channel một cách máy móc.

# **4. Có một vấn đề lớn hơn: Cosine 0.751 không đo decoder compatibility**

Đây là điểm tao muốn mày đặc biệt chú ý.

Mapping đang tối ưu:

\\[ \min\_{W,b}\\|XW+b-Y\\|^2 \\]

Tức là cố gắng làm cho mapped representation giống Flan encoder representation.

Nhưng decoder không trực tiếp tối ưu cosine hoặc MSE.

Nó cần:

\\[ P(y_t\mid y\_{\<t},H\_{\text{mapped}}) \\]

Nghĩa là xác suất sinh token tiếp theo khi nhìn vào mapped hidden states.

Hai mục tiêu này khác nhau.

Một mapping có MSE thấp chưa chắc tạo ra translation tốt nhất.

Ngược lại, một mapping có cosine thấp hơn đôi chút vẫn có thể tạo ra translation tốt hơn nếu nó giữ được các đặc trưng quan trọng cho decoder.

Đây là lý do tao nghĩ nghiên cứu của mày có thể cần chuyển từ:

**Representation similarity objective**

sang thử nghiệm:

**Decoder-aware representation alignment.**

Nhưng chưa nên vội làm điều đó. Cần có baseline chẩn đoán trước.

# **5. Tao đề xuất một experiment có thể trả lời gần như toàn bộ tranh luận này**

Thay vì tiếp tục tranh luận encoder hay decoder yếu, hãy dùng cùng một decoder, cùng một tập câu và cùng một target để kiểm tra.

| Test | Encoder states                     | Decoder |
| ---- | ---------------------------------- | ------- |
| A    | Flan native                        | Frozen  |
| B    | ViT5 mapped                        | Frozen  |
| C    | ViT5 mapped                        | LoRA    |
| D    | ViT5 mapped, oracle-like reference | Frozen  |

Trong đó D cần thiết kế cẩn thận: sử dụng representation target từ Flan encoder làm tham chiếu, không phải một oracle có thể truy cập thông tin tương lai trong quá trình generation.

Các kết quả sẽ cho biết:

- A tốt, B kém: mapped representation chưa tương thích với decoder.
- B kém, C tốt: decoder adaptation có ích.
- B và C đều kém: cần xem lại mapping hoặc cách alignment.
- A và B đều kém: có thể decoder hoặc thiết lập generation đang có vấn đề.

Tao còn muốn thêm một phép đo trực tiếp:

**Teacher-forced NLL trên cùng target.**

Thay vì chỉ nhìn BLEU sau khi greedy decoding, đo:

\\[ L=-\sum_t\log P(y_t\mid y\_{\<t},H) \\]

Nếu cùng một decoder, cùng target:

- Native NLL thấp.
- Mapped NLL cao.

Thì có bằng chứng decoder thực sự khó sử dụng mapped states.

Nếu NLL gần nhau nhưng BLEU chênh lệch lớn, ta cần điều tra thêm decoding và lỗi tích lũy autoregressive.

# **6. Quan điểm hiện tại của tao**

Tao sẽ không gọi encoder là tốt và decoder là yếu nữa.

Tao sẽ nói chính xác hơn:

| Thành phần     | Điều đã biết                         | Điều chưa biết                          |
| -------------- | ------------------------------------ | --------------------------------------- |
| ViT5 encoder   | Tạo representation tiếng Việt        | Giữ đủ thông tin cho translation không? |
| Linear mapping | Học được correspondence rõ rệt       | Đủ tương thích với decoder chưa?        |
| Flan decoder   | Copy được gần như tuyệt đối sau LoRA | Đọc mapped states tốt đến đâu?          |
| Translation    | Mapped tốt hơn native                | Nút thắt chính nằm ở đâu?               |

**Nếu phải chọn việc tiếp theo, tao sẽ chưa train thêm epoch. Tao sẽ đo teacher-forced NLL của native và mapped trên cùng decoder trước.**

Và tao muốn hỏi mày đúng một điều kỹ thuật để tránh phân tích sai:

Trong Phase 1, decoder được train với cả hai channel `native` và `mapped` xen kẽ trong cùng quá trình training, đúng không? Hay thực tế hai channel được train theo hai giai đoạn khác nhau?

Thông tin này ảnh hưởng rất lớn đến việc diễn giải kết quả 6.28 so với 3.85 BLEU.
---

Haha, đúng, bro đang chạm vào một điểm rất quan trọng. :))

Tui nghĩ cần chỉnh lại cách đặt vấn đề một chút: **dịch tốt chính là sinh ra phân phối token đích phù hợp với câu nguồn**, chứ không phải tái tạo hidden state của Flan-T5 cho giống nhất có thể.

Nhưng có một khác biệt quan trọng giữa hai ý:

- **Đúng phân phối token:** Với câu nguồn \\(x\\), mô hình cần học \\(P(y\mid x)\\), tức xác suất của các chuỗi tiếng Anh hợp lý.
- **Giống hidden state:** Mapping cần tạo ra biểu diễn gần với hidden state mà encoder Flan-T5 tạo ra khi đọc câu tiếng Anh tương ứng.

Hai thứ có liên quan, nhưng không tương đương.

## **1. Linear mapping đang tối ưu một mục tiêu gián tiếp**

Trong Phase 1, bro học:

\\[ H\_{\text{mapped}} = H\_{\text{ViT5}}W+b \\]

với objective:

\\[ \mathcal{L}\_{map} = \\|H\_{\text{mapped}}-H\_{\text{Flan}}\\|^2 \\]

Đây là cách ép hai không gian biểu diễn gần nhau.

Nhưng mục tiêu dịch thực sự là:

\\[ \mathcal{L}\_{MT} = -\sum_t \log P(y_t\mid y\_{\<t},H\_{\text{mapped}}) \\]

Tức là làm cho decoder dự đoán đúng token tiếng Anh tiếp theo.

**Điểm thú vị:** Không nhất thiết phải tái tạo đúng hidden state Flan-T5. Chỉ cần hidden state sau mapping chứa đủ thông tin và tương thích với decoder để sinh đúng phân phối token đích.

Thậm chí, một mapping có MSE cao hơn vẫn có thể dịch tốt hơn nếu nó giữ lại những thông tin decoder cần.

## **2. Có nên LoRA hoặc fine-tune trực tiếp trên dataset dịch?**

Tui nghĩ đây chính là hướng đáng thử nhất.

Có thể chia thành ba thí nghiệm:

| Thí nghiệm             | Mapping                  | Decoder              |
| ---------------------- | ------------------------ | -------------------- |
| A — Baseline           | Linear mapping cố định   | Flan decoder cố định |
| B — Decoder adaptation | Linear mapping cố định   | LoRA decoder         |
| C — Joint adaptation   | Linear mapping trainable | LoRA decoder         |

Với C, loss trực tiếp là translation cross-entropy:

\\[ \min\_{W,b,\theta\_{\text{LoRA}}} -\sum_t \log P\_{\theta\_{\text{LoRA}}} (y_t\mid y\_{\<t},H\_{\text{ViT5}}W+b) \\]

Tức là mapping không còn phải đoán hidden state nào giống Flan-T5 nhất. Nó được phép học cách biến đổi vector sao cho decoder dịch tốt nhất.

Có thể đóng băng toàn bộ ViT5 encoder và Flan backbone, chỉ cập nhật \\(W,b\\) và LoRA. Như vậy vẫn giữ được ý tưởng vector-space transfer ban đầu.

## **3. Về BLEU 40 — bro nói đúng :))**

BLEU 40+ là benchmark rất đáng kể

Bảng bro đưa cho thấy trên PhoMT:

- Transformer-base: 37.19 Vi→En.
- mBART: 39.78 Vi→En.
- EnViT5-base: 39.57 Vi→En.
- MTet + PhoMT: 40.57 Vi→En.
- Google/Bing Translator: khoảng 35.7 Vi→En.

Trong khi Phase 1 của bro đạt BLEU 6.28.

Nhưng cần công bằng khi so sánh: các kết quả trong bảng dùng benchmark, dữ liệu huấn luyện, pretrained model và quy trình fine-tune khác nhau. Không thể lấy BLEU 40 làm ngưỡng bắt buộc cho một proof-of-concept vector mapping.

Dù vậy, khoảng cách 6.28 → 40 là quá lớn để chỉ giải thích bằng chuyện cần thêm một chút training. Nó cho thấy pipeline hiện tại chưa khai thác được khả năng dịch của mô hình.

## **4. Tui sẽ ưu tiên thí nghiệm nào?**

Tui đề xuất một thí nghiệm rất có giá trị khoa học:

**Frozen ViT5 + Linear Mapping + LoRA Flan Decoder**

**ViT5 Encoder**

Frozen

**Linear Mapping W, b**

Trainable

**Flan-T5 Decoder**

Frozen backbone + trainable LoRA

**English token distribution**

Optimize translation cross-entropy

**So sánh với hai baseline:**

- Mapping cố định + decoder LoRA.
- Mapping trainable + decoder LoRA.

Một lưu ý: nên có thêm baseline Flan-T5 native encoder + cùng decoder LoRA, dùng cùng dữ liệu và training budget. Nếu không, ta khó biết bottleneck nằm ở mapping hay ở khả năng dịch của decoder.

**Và đây mới là câu hỏi nghiên cứu tui thấy hấp dẫn:**

> Liệu ta có thể chuyển hidden states từ một encoder tiếng Việt sang không gian đầu vào của một decoder khác, rồi dùng translation loss để học phép chuyển đổi mà không cần fine-tune encoder?

Nếu kết quả tốt, đó là bằng chứng mạnh hơn nhiều so với việc chỉ đạt cosine cao giữa hai không gian.

Tui không nghĩ nên bỏ linear mapping. Tui nghĩ nên cho nó cơ hội được tối ưu theo đúng mục tiêu cuối cùng: **dịch**, thay vì chỉ bắt chước hidden states.
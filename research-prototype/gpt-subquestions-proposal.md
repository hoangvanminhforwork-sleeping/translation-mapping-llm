A. Representation thực sự chứa cái gì?
Đây là câu hỏi tui thích nhất.

Ông đang giả định rằng:
Nếu hai model hiểu cùng một nội dung, representation của chúng có thể được align thông qua một mapping function.

Nhưng có một câu hỏi nằm trước cả mapping:

Hai model có thực sự encode cùng một loại information vào representation của chúng không?

Ví dụ, xét câu:

"Trâu gác bếp là một món ăn truyền thống của người Thái ở vùng Tây Bắc Việt Nam."

Model tiếng Việt có thể encode:
- Trâu là nguyên liệu chính.
- Gác bếp là một phương pháp chế biến.
- Đây là món ăn truyền thống.
- Có liên hệ với một cộng đồng văn hóa cụ thể.
- Có liên hệ với một vùng địa lý cụ thể.
Trong khi model tiếng Anh có thể hiểu:
- Buffalo meat.
- Smoked or dried meat.
- Traditional food.
- Northwest Vietnam.
Nhưng liệu hai representation có giữ được những thông tin đó với mức độ tương đương?

Tui đề xuất một research question:

RQ: To what extent do cross-lingual representation mappings preserve different types of semantic information?

Có thể chia thành các subquestions:
- Lexical information: Mapping có giữ được nghĩa từ vựng không?
- Semantic information: Mapping có giữ được nội dung câu không?
- Cultural information: Mapping có giữ được thông tin văn hóa không?
- Factual information: Mapping có giữ được các sự kiện cụ thể không?
- Relational information: Mapping có giữ được quan hệ giữa các thực thể không?
Điểm thú vị: Có thể mapping đạt cosine similarity rất cao nhưng lại mất một số cultural information quan trọng.

Đây sẽ là một cách rất tốt để chứng minh rằng MAE, MSE hay cosine similarity chưa đủ để đánh giá representation alignment.

B. Mapping có bảo toàn information không, hay đang tạo ra information mới?
Câu hỏi này liên quan trực tiếp tới loss mà ông đang nghiên cứu.

Giả sử:
\[
h_s \xrightarrow{f_\theta} \hat h_t
\]

Trong đó:
- \(h_s\): Source representation.
- \(f_\theta\): Mapping function.
- \(\hat h_t\): Predicted target representation.
- \(h_t\): Ground-truth target representation.
Ông có thể đo:
\[
L_{\text{representation}} =
\|\hat h_t-h_t\|^2
\]

Nhưng tui muốn ông nghĩ thêm một vấn đề:

Nếu mapping tạo ra một representation có khoảng cách rất nhỏ so với ground truth, liệu model đích có thực sự generate ra cùng một nội dung không?

Và ngược lại:

Nếu representation có khoảng cách tương đối lớn nhưng vẫn generate ra câu có cùng ý nghĩa thì sao?

Đây là hai vấn đề hoàn toàn khác nhau.

Tui đề xuất:

RQ: How does representation alignment error propagate into downstream token generation?

Các câu hỏi nhỏ:
1. Representation error có tương quan với token prediction error không?
2. Những chiều embedding nào có ảnh hưởng lớn nhất tới kết quả generation?
3. Một sai lệch nhỏ ở representation có thể khiến model generate ra một từ hoàn toàn khác không?
4. Có tồn tại một ngưỡng representation error mà sau đó generation quality suy giảm đột ngột không?
5. Sai lệch representation ảnh hưởng đến token đầu tiên hay tích lũy dần qua các bước autoregressive generation?
Câu số 5 khá hay.

Ông có thể nghiên cứu error propagation theo từng bước generate:
\[
e_1 \rightarrow e_2 \rightarrow e_3 \rightarrow \cdots \rightarrow e_n
\]

Trong đó \(e_i\) là sai lệch ở bước generation thứ \(i\).

Không nhất thiết phải dùng các thuật toán dự đoán chuỗi kiểu chứng khoán đâu. Trước tiên, hãy đo error propagation thực nghiệm. Sau đó mới xem có cần một mô hình dự đoán hay không.

C. Mapping có thực sự đối xứng không?
Ông đang có:
\[
English \rightarrow Vietnamese
\]

Và:
\[
Vietnamese \rightarrow English
\]

Nhưng ông có một giả định khá thú vị mà chưa nhắc tới:

Nếu mapping từ English sang Vietnamese hoạt động tốt, mapping ngược lại có hoạt động tốt không?

Giả sử:
\[
f_{EV}: H_E \rightarrow H_V
\]

\[
f_{VE}: H_V \rightarrow H_E
\]

Ta có thể kiểm tra:
\[
f_{VE}(f_{EV}(h_E)) \approx h_E
\]

Đây là một dạng cycle consistency.

Research questions:
- Does cross-lingual representation mapping exhibit directional asymmetry?
- Does mapping in one direction preserve more semantic information than mapping in the opposite direction?
- Can cycle consistency improve representation alignment?
- Does the mapping preserve cultural concepts equally well in both directions?
Tui đặc biệt thích câu cuối.

Ví dụ, dịch một khái niệm Việt Nam sang tiếng Anh có thể mất thông tin văn hóa. Nhưng khi dịch ngược lại, model có thể không khôi phục được thông tin ban đầu.

Đây là một vấn đề rất sát với research objective của ông.

D. Mapping có phụ thuộc vào loại câu không?
Một mapping được train trên các câu thông thường có thể hoạt động tốt với:
- "I eat rice."
- "She likes cooking."
- "This is a traditional dish."
Nhưng liệu nó có hoạt động tốt với:
- "Thịt trâu gác bếp."
- "Bánh chưng."
- "Mắm tôm."
- "Bún đậu mắm tôm."
Hay những câu có nhiều thông tin văn hóa?

E. Semantic information có tập trung ở một layer không?
Ông đang tìm kiếm:
At which layer does the model contain the most semantic information?

Tui nghĩ câu hỏi này cần được tinh chỉnh một chút.

Không chắc tồn tại một layer duy nhất chứa nhiều semantic information nhất.

Các layer có thể encode những loại information khác nhau.

Ví dụ, một giả thuyết có thể kiểm chứng:
- Early layers: Lexical và positional information.
- Middle layers: Syntactic và compositional information.
- Later layers: Contextual và task-related information.
Đây không phải quy luật cố định cho mọi kiến trúc, nên cần thực nghiệm.

F. Layer có tương ứng với nhau giữa hai model không?
Ví dụ:
- Flan-T5 Base: Layer 6.
- ViT5 Base: Layer 6.
Liệu hai layer này có thực sự encode information ở mức tương đương không?

Không có gì đảm bảo cả.

Tui đề xuất:

RQ: Is cross-lingual alignment dependent on corresponding layer depths, or can non-corresponding layers provide better alignment?

Thay vì chỉ thử:
\[
X=Y
\]

Hãy thử:
\[
(X,Y)\in
\{(1,1),(1,2),(1,3),...,(12,12)\}
\]

Tức là tạo một layer-pair alignment matrix.

Minh họa về mặt thiết kế thí nghiệm, không phải kết quả thực nghiệm.







bair.berkeley.edu


Sau đó đánh giá từng cặp bằng:
- Representation similarity.
- Semantic probing.
- Generation quality.
- Cultural information preservation.
Một heatmap sẽ giúp ông nhìn thấy sweet spot rõ hơn nhiều so với chỉ thử một vài cặp layer tùy ý.

H. Có tồn tại một cultural information loss không?
Ví dụ:
Original	Translation
Trâu gác bếp	Smoked buffalo meat
Bánh chưng	Vietnamese sticky rice cake
Mắm tôm	Fermented shrimp paste

Các bản dịch trên có thể truyền tải được một phần thông tin.

Nhưng chúng không nhất thiết truyền tải được:
- Nguồn gốc văn hóa.
- Cách chế biến đặc trưng.
- Ngữ cảnh sử dụng.
- Vai trò trong đời sống.
- Ý nghĩa biểu tượng.
Tui đề xuất:

RQ: To what extent does conventional machine translation preserve culturally embedded information in Vietnamese food terminology?

Và câu tiếp theo:

Can dynamic fact insertion recover cultural information lost during translation?

Đây chính là nơi RAG của ông phát huy tác dụng.

Tuy nhiên, có một vấn đề thú vị:

RAG có thực sự cần thiết cho tất cả các thuật ngữ văn hóa không?

Ví dụ:
- "Bánh mì": Có lẽ không cần giải thích quá nhiều.
- "Mắm tôm": Có thể cần giải thích nguyên liệu và cách chế biến.
- "Trâu gác bếp": Có thể cần giải thích cả phương pháp chế biến, nguồn gốc và đặc điểm hương vị.
Tức là:

I. How much cultural explanation does each food-related expression require?

Đây có thể trở thành một bài toán phân loại:
Cultural explanation levels
Level 0 — No explanation
Low

Món ăn đã quen thuộc với độc giả mục tiêu.

Example: Bánh mì → Banh mi


Level 1 — Brief explanation

Cần giải thích một đặc điểm cơ bản.

Example: Mắm tôm → Fermented shrimp paste


Level 2 — Cultural context

Cần giải thích nguồn gốc, cách chế biến hoặc ngữ cảnh văn hóa.

Example: Trâu gác bếp → Smoked and dried buffalo meat traditionally prepared by Thai communities in Northwest Vietnam.


Level 3 — Extended explanation

Cần giải thích cả ý nghĩa xã hội, lịch sử hoặc biểu tượng.

Example: Những món ăn gắn với nghi lễ, lễ hội hoặc phong tục địa phương.


Ông có thể nghiên cứu xem hệ thống có tự xác định được explanation level phù hợp không.

Đây là một câu hỏi có giá trị ứng dụng rất rõ

K. Người đọc thực sự hình dung món ăn như thế nào?
Đây là câu hỏi tui cực kỳ khuyến khích ông cân nhắc.

Ông đang lấy perplexity và probe để đánh giá model.

Nhưng mục tiêu cuối cùng của ông là người đọc không hình dung sai về món ăn.

Hai thứ đó không hoàn toàn tương đương.

Ví dụ:
- Translation A: Smoked buffalo meat.
- Translation B: Traditional Northwest Vietnamese buffalo meat, slowly smoked and dried above a kitchen fire.
Câu B dài hơn, nhưng liệu người đọc có hình dung chính xác hơn không?

Đây là một câu hỏi có thể đánh giá bằng human evaluation.

RQ: Does culturally informed translation improve readers' understanding and mental representation of Vietnamese food concepts?

Có thể đo bằng:
1. Ingredient identification: Người đọc xác định đúng nguyên liệu không?
2. Cooking method identification: Người đọc hiểu đúng phương pháp chế biến không?
3. Cultural association: Người đọc nhận biết đúng nguồn gốc văn hóa không?
4. Expectation accuracy: Người đọc hình dung đúng đặc điểm món ăn không?
5. Misconception rate: Tỷ lệ người đọc hình dung sai giảm bao nhiêu?
Cái số 4 và 5 rất đáng chú ý.

Nó đánh giá đúng vấn đề mà ông muốn giải quyết, thay vì chỉ đánh giá câu dịch có giống reference translation hay không.

O. Mapping có phụ thuộc vào hidden dimension không?
Flan-T5 Base và ViT5 Base có hidden dimension tương ứng là 768; các phiên bản Large tương ứng là 1024.

Nhưng nếu sau này ông muốn mapping giữa các model khác kích thước:
\[
H_E \in \mathbb{R}^{n\times d_E}
\]

\[
H_V \in \mathbb{R}^{m\times d_V}
\]

Khi \(d_E \neq d_V\), mapping cần xử lý khác biệt dimensionality.

RQ: How does hidden dimensionality mismatch affect cross-lingual representation alignment?

Có thể thử:
- Linear projection.
- Two-layer MLP.
- Low-rank mapping.
- Dimension reduction.
- Dimension expansion.

Mapping có cần thiết phải học toàn bộ không?
Ông đang định train một mapping function:
\[
f_\theta:H_E\rightarrow H_V
\]

Nhưng thử nghĩ thế này:

Nếu chỉ một số chiều trong representation thực sự quan trọng cho cultural information thì sao?

Ví dụ:
- 768 dimensions.
- Nhưng chỉ 100 dimensions đóng góp đáng kể vào việc bảo toàn một số loại semantic information.
- Những dimension còn lại có thể chứa information ít liên quan tới nhiệm vụ.
Từ đó có câu hỏi:

RQ: Is cross-lingual representation alignment concentrated in a subset of latent dimensions?

Có thể thử:
- Feature importance analysis.
- Dimension ablation.
- PCA.
- Sparse linear mapping.
- Low-rank mapping.
Một câu hỏi mở rộng:

Can a smaller mapping function preserve cultural semantics as effectively as a full-dimensional mapping?

Cái này rất hợp với mục tiêu lightweight của ông.

Tuy nhiên, đừng mặc định mỗi dimension tương ứng với một khái niệm độc lập. Representation của neural network thường có tính distributed.

Level 1 — Representation
Technical

- MSE / MAE
- Cosine similarity
- CKA / SVCCA
- Layer-wise alignment matrix
- Cycle consistency


Level 2 — Semantic information
Probing

- Food-related classification
- Ingredient identification
- Cooking method classification
- Cultural origin classification
- Semantic relation prediction


Level 3 — Generation
NLP

- BLEU / chrF
- COMET
- BERTScore
- Token-level negative log-likelihood
- Perplexity (with appropriate controls)
- Hallucination / factual consistency


Level 4 — Cultural understanding
Application

- Cultural information preservation
- Explanation adequacy
- Misconception rate
- Reader comprehension
- Human preference evaluation

Câu hỏi phụ đề xuất bởi ChatGPT:

- Câu hỏi chính: How can cross-lingual representation alignment and dynamic cultural fact insertion improve the translation and cultural understanding of Vietnamese food-related expressions?

Tui chọn câu này vì nó bao quát được cả ba thành phần:
- Translation: Dịch đúng.
- Cultural preservation: Giữ được thông tin văn hóa.
- Cultural understanding: Giúp người đọc hiểu đúng.

- Câu hỏi nhỏ hơn 1: Information (cả về culture lẫn semantic) trong representation có bị mất hay không?

- Câu hỏi nhỏ hơn 2: Cultural translation strategy?
+ Khi nào nên borrowing (giữ nguyên ko dấu), literal translation, descriptive translation, translation và explanation?

- Câu hỏi nhỏ 3: Dynamic fact
+ RAG có tìm dc thông tin về "văn hóa" không?
+ Giải thích dài bao nhiêu là phù hợp?

- Câu hỏi nhỏ 4: Nhận định của máy và của người correlate tới mức nào?
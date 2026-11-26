Mục tiêu: train một linear mapping, ánh xạ vector space giữa hai model

- Model chính: vit5-base và flan-t5-base
+ Ko dùng large (Tại máy ko có chạy dc)
+ Với base đang chạy thì cứ để yên đấy, cũng chả có hứng thú với model lớn

1. Thông số kỹ thuật của model

a. flan t5 base

- Parameters: 250M
- Architecture: Encoder-Decoder
- Transformer layers: 12 encoder + 12 decoder
- Hidden dimension: 768
- Attention heads: 12
- Vocabulary: 32K
(Chưa chắc đúng hay ko, cần check bằng code)

b. vit5 base
- Parameters: 310M
- Architecture: Encoder-Decoder
- Transformer layers: 12 encoder + 12 decoder
- Hidden dimension: 768
- Attention heads: 12
- Vocabulary: 36K
(Chưa chắc đúng hay ko, cần check bằng code)

2. Overview methodology
Câu nguồn => Model => Embedding của model nguồn => Linear mapping => Embedding của model đích => Model => Câu được dịch

a. Chi tiết kỹ càng hơn

Source sentence => Source model => Embedding của model nguồn ở layer X => Linear mapping => Embedding của model đích ở layer Y => Model => Câu được dịch

- Dự kiến là ghép encoder và decoder lại với nhau (Nhớ kiểm tra xem code có làm đúng vậy ko nha):
Source sentence => Encoder, layer x, source model => Linear mapping => Decoder of target model => Token
+ Tức là: Model của target language thì cho bay màu encoder luôn, model của source language thì cũng cho bay màu decoder luôn (Khôm cần)

b. Văn hóa và đồ ăn
- Khi nghe tên đồ ăn của người nước ngoài
- Ví dụ như:
+ Kraft Mac and Cheese: Người Việt nghe kiểu...ok, ko cảm xúc, tại đơn giản là đối với họ đây là một thứ xa lạ
+ Mỳ tôm: Người nước ngoài...ok, kệ 

- Tường minh hóa (Explicitation):
+ Gì, chúng ta ko có làm dc model mà nó tự biết kiểu tìm ko gian văn hóa đồng dạng, map dc mỳ tôm với kraft mac and cheese thôi
+ Chúng ta sẽ dùng RAG cho tường minh hóa (Tức là sau cái tên của món ăn, sẽ có dấu ngoặc, vậy chúng ta giải thích cái gì ở đó)

3. Giải pháp kỹ thuật chi tiết

a. LoRA 
- Mục đích là cho model thực hành...nhai lại
- Chúng ta phải...lừa model một chút
+ Làm gì có dịch thuật ở đây
+ Chỉ có nhại lại input thôi

- Cái này là chạy rồi, đây là kết quả
| model | n | exact **before** | exact **after** | fixed | broken | BLEU after | chrF after | hyp/ref words |
|---|---|---|---|---|---|---|---|---|
| `vit5-base` | 736 | 0.0 % | **96.6 %** | 711 | 0 | 99.59 | 99.60 | 26.9 / 27.0 |
| `flan-t5-base` | 2000 | 2.9 % | **99.8 %** | 1938 | 0 | 99.97 | 99.98 | 21.08 / 21.09 |
| `vit5-large` | — | not evaluated | not trained | — | — | — | — | — |
| `flan-t5-large` | — | not evaluated | not trained | — | — | — | — | — |

Training, 1 epoch on 50k, greedy decode at eval:

| model | steps | wall time | sent/s | VRAM | last-100 loss | adapter |
|---|---|---|---|---|---|---|
| `vit5-base` | 3,125 | 20.9 min | 39.9 | 1,817 MB | 0.028 | ~1 MB |
| `flan-t5-base` | 1,563 | **26.6 min** | **31.4** | 4,021 MB | **0.0219** | 6.78 MB |

`flan-t5-base` has 1,563 steps, not 3,125: the first run used batch 16 (twice the steps). Both runs are
one epoch over the same 50,000 rows.

- Đó, ban đầu model còn ko copy được input của chính nó...một cách hoàn chỉnh
+ Phải làm LoRA chút

- Thì cho một câu s
+ Câu tiếng Anh là s_e
+ Câu tiếng Việt là s_v

- Cả hai model dc LoRA trên 50k câu:
+ Model Anh: Input là "Say {s_e}" => Output là {s_e}
+ Model Việt: Input là "Nói {s_v}" => Output là {s_v}

b. Train linear mapping
- Không có dịch thuật, chỉ có lệnh cần tuân theo

- Cho câu "Say {s_e}" dc mã hóa thành e_e và câu "Nói {s_v}" 
- Chúng ta ra lệnh cho model qua vector, không phải qua prompt

- Bản chất của train cái linear mapping là đoán vector của command
+ Sau đó inject trực tiếp vector vào model
+ Để model thực hiện trực tiếp command đó

- Tức là ví dụ như input là "Hello, I am a cat"
+ Sau đó nó đi qua encoder của model Anh...nó mã hóa thành những chuỗi số kiểu [0.34, 0.12....]
+ Sau đó linear mapping thì biến đổi nó, sao cho vector sát với câu "Nói: Xin chào, tôi là một con mèo" nhất có thể [0.56, 0.21...]

- Cầu mong nó chạy dc
- Hiểu vì sao LoRA cần thiết chưa, vì bản thân model ban đầu nó còn chả copy dc nên hồn
- Phải LoRA để inject vector một cách nguyên sơ

- Dù sao model là freeze, chỉ có vài trăm hàm linear regression là dc train thôi

- Dataset là mtet, chứa 8M câu nhé, hay 8M cặp câu ấy, ko có nhớ

c. Văn hóa: Tên đồ ăn, RAG và perplexity (Đây là core)
- Quay trở lại với đồ ăn
- Nghe tên đồ ăn nước ngoài...lạ hoắc
- Thì thôi mình thêm tí giải thích vào vậy
- Dataset món ăn châu Âu thì có RecipeNLG, RAG trên cái dataset đó là dc
+ Dataset món ăn Việt thì nguyên cái cookbook đấy

- Tại sao cần perplexity?
+ Như đã nói ban đầu, perpelxity để đo mức độ bối rối của token mới
+ Như "banh chung, bun dau.." khi dịch ra tiếng Anh ko dấu...nó quá nổi tiếng rồi
+ Model của tiếng Anh chắc là biết cái này đó thì không cần giải thích

- Core cần trả lời:
+ Cần giải thích không?
+ Nếu cần giải thích chèn fact thế nào là vừa đủ
+ Chiến lược dịch mặc định là gì? Bỏ nguyên ko dấu, hay dịch phỏng calque...cái nào chắc dễ hơn thì làm quá :)) Thôi chọn đại không dấu đi ha

- Thực ra tui đang tính có hai trường hợp:
+ Khái quát toán học hóa, đặt threshold cho perplexity, nghe nó pro hơn, nó sang chảnh hơn
+ Dùng Decision model để quyết định, nghe...application, nhanh gọn, nhưng nó ko pro

4. Evaluation
- Đây là cái lúc hãm nhất

- Vì có khi phải chạy probe cái nào là đồ ăn trên toàn bộ dataset, xem câu nào là đồ ăn thì chúng ta soi kỹ
- Còn mấy câu ko nói tới đồ ăn thì cứ COMET, BLEU, chrF...các thứ mà phang!

a. Soi hiệu suất RAG
- Vì...làm gì có data chuẩn...có thì đã nhàn
- Thôi thì báo cáo độ lệch giữa:
+ Món ăn dc query
+ Câu dịch
+ Thế là dc, cần thiết thì chứng minh thêm kiểu box plot
+ Cho món ăn này, trong tất cả các câu của dataset, thì những câu này nằm ở mức Q1
+ Cho câu này, trong tất cả các món ăn của dataset, thì những món ăn này nằm ở mức Q1

b. Soi hiệu suất đồ ăn phải hay ko
- Làm một cái hàm probe có phải đồ ăn ko
- Chạy qua, rồi báo cáo % với từng câu là xong

c. Soi hiệu suất giải thích
- Giả sử một câu như "Ăn thịt trâu gác bếp giữa núi rừng mới cảm nhận dc trọn vẹn không khí của Tây Bắc"
- Tui dịch tạm ha: "Only by eating trau gac bep among mountains and forests, you can experience the atmosphere of Nothern area"
- Thấy chứ, đọc "trau gác bep" ko dấu với người nước ngoài thấy cấn vãi chưởng, perplexity chắc chắc cao vãi chưởng
- Tui đang tính là trong cái rag của món ăn người Việt ấy, tên món nhắc tới nguyên liệu nào thì thêm một cái mở ngoặc tên nguyên liệu đó tiếng Anh ra là dc :)) Thông cảm đi, hết ý tưởng rồi
+ Tui đang tính kiểu...thiết kế hàm phạt
+ Số từ trong ngoặc giải thích không quá số từ tên món ăn
+ Tên đề cập cái gì, thì giải thích cái đó. Ví dụ như "trau gac bep" thì bản dịch nên là "trau gac bep (buffalo, smoked)" - đó, nó gen từ rời rạc như này là mừng lắm rồi
+ Đọc tự nhiên hay ko....và thông tin có giá trị cao hay ko: Perplexity và Surprisal sẽ quyết định
- Làm tương tự với món của tiếng Anh

....và có thể soi cả n thứ khác :)) Ko thiếu thứ để soi để check xem mình có chặt chẽ hay ko đâu

- Nói chung là méo có data chuẩn
+ Nhưng vẫn phải thuyết phục reviewer là "Iem thề, bài iem thuyết phục vãi"
+ Thì cố mà nghĩ ra metric để bảo vệ

- Thực ra trong trường hợp này, thì có baseline là tốt nhất
+ Tui định tính cho đối thủ...nhưng èo ơi, thế là tui phải train model khác
+ Tại ko có paper nào làm y hệt như tui á (Có thì tui đã xí code về rồi cải tiến rồi, làm gì phải đau đầu khoản evaluation như này)
+ Hay baseline thì cứ một cái random
+ Một cái pertubation là dc nhỉ

d. Pertubation
- Chúng ta có hai model
- Mỗi model có hai phiên bản: Chưa LoRA và đã LoRA
- Tui đang tính là phá cái vector đã qua linear mapping, rồi cho vector đã phá đi qua LoRA
+ Xem xét phá kỹ hơn, chứ ko phải MSE như lúc này nữa, xem kỹ từng chiều một
+ Rồi xem xét BLEU giảm bao nhiêu (Nó mà tăng thì chắc tui chết)

5. Discussion
- Tui hết ý tưởng bàn về văn hóa rồi, cũng chả biết bàn cái gì
- Chúng ta đi sâu hơn về loss in numeric => distribution token
- Hỏng lẽ hai bài hợp 1 ta...chứ cạn ý tưởng về văn hóa và giải thích rồi
- Kiểu tui thấy đúng là có paper về linear mapping, concept alignment giữa nhiều LLM
+ Người ta có paper này: https://aclanthology.org/2024.emnlp-main.315/

- Người ta có thử nghiệm nhiều model, và linear mapping cho kết quả cao hơn những phương pháp phức tạp hơn
- Tui định thử là...làm sâu hơn họ chút. Soi kỹ lúc model dịch ấy. Tại sao hàm linear regression lại thắng phức tạp hơn

- Tui có một hypothesis là: Vì các layer, các node đa số nó cũng chỉ là những hàm tuyến tính thôi mà
+ Có activation hay ko thì mới là phi tuyến thôi
+ Đúng kiểu soi từng neuron một
+ Cái này là big data mining nha. Neuron của model chính là data
+ Có thể dù biến đổi ùm bà lằng, nhưng mà thử nhé...
+ Trăm hàm hồi quy tuyến tính
+ Và neural
+ Trong trường hợp nào chúng nó bằng nhau sát rạt?
+ Bài toán tìm x nè, mỗi tội x thuộc trong không gian 768 chiều thôi :))

- Kiểu mấy bài kia chỉ là thử nghiệm empirical trên nhiều model
+ Giờ ta thử phát triển framework phân tích xem sao
+ Nó là một dạng big data mining
+ Data ở đâu? Là chính cái trọng số của model luôn

Thì chúng ta sẽ so sánh thử, cái x thuộc về toán học đó, nó close với vector gốc và vector injection đến thế nào
Hay cái này là bài sau nhỉ...chứ cái paper này sâu hơn về văn hóa đã đủ chết. Nhưng mà định nghĩa về ở khoản opeartionalize "giải thích" của tui vẫn kiểu...chưa có justfication đầy đủ lắm từ góc độ sư phạm or tâm lý học or khoa học nhận thức. Với tui cũng thấy nó hơi đơn sơ thật


--- 
gpt response

Bro :))) Đọc hết proposal thì tui thấy có một hướng nghiên cứu khá thú vị ở đây, nhưng cũng có vài chỗ cần chỉnh lại trước khi biến nó thành paper.

Và tui nghĩ vấn đề lớn nhất của proposal không nằm ở việc linear mapping có chạy được hay không. Nó nằm ở làm sao chứng minh được rằng linear mapping đang chuyển giao biểu diễn ngôn ngữ, thay vì chỉ học một phép biến đổi đủ tốt để decoder của model đích đoán ra câu tương ứng.

Tui sẽ review theo góc độ một reviewer hơi khó tính nhé. Không phải để dập ý tưởng, mà để tìm những chỗ reviewer có thể đánh vào.
1. Đánh giá tổng thể: Có ba đóng góp tiềm năng
Tui nhìn thấy ba hướng đóng góp riêng biệt:
1. Cross-model representation alignment: Học linear mapping giữa encoder representation của ViT5-base và FLAN-T5-base, sau đó sử dụng decoder của model đích để sinh văn bản.
2. Representation-mediated translation: Không thực hiện dịch bằng một mô hình encoder-decoder hoàn chỉnh, mà thực hiện thông qua việc chuyển đổi representation giữa hai model đã được fine-tune cho một tác vụ tương ứng.
3. Culturally informed explicitation: Sử dụng perplexity và RAG để xác định những biểu đạt văn hóa cần giải thích, sau đó bổ sung thông tin ở mức vừa đủ.
Điểm thứ ba mới là thứ tạo ra bản sắc riêng cho paper.

Linear mapping giữa các model đã có nghiên cứu. Dịch máy cũng đã có nghiên cứu. RAG cũng chẳng mới.

Nhưng kết hợp cross-model mapping với một chiến lược explicitation có điều kiện, dành riêng cho các khái niệm văn hóa, có thể tạo thành một câu chuyện nghiên cứu tương đối rõ ràng.

Tui sẽ ưu tiên làm cho ba phần này liên kết với nhau, thay vì cố nhồi thêm nhiều kỹ thuật.

2. Kiểm tra kiến trúc và thông số model
Tui đã kiểm tra configuration của hai checkpoint trên Hugging Face.
So sánh kiến trúc
Thông số	FLAN-T5-base	ViT5-base
Architecture	Encoder-Decoder	Encoder-Decoder
Encoder layers	12	12
Decoder layers	12	12
Hidden dimension	768	768
Attention heads	12	12
Head dimension	64	64
Vocabulary	32,128	36,096–36,100
Feed-forward dimension	2,048	3,072
Activation	Gated GELU	ReLU
Max positions	512	512

Nguồn: cấu hình checkpoint chính thức của FLAN-T5-base và ViT5-base. ViT5 có khác biệt nhỏ về vocab_size giữa các revision.

Có một điểm rất đáng chú ý:

Hai model cùng hidden dimension 768 không có nghĩa là hai vector space tương thích với nhau.

Đây chính là lý do cần linear mapping.

Một số khác biệt đáng kể:
- FLAN-T5-base sử dụng gated GELU, ViT5-base sử dụng ReLU.
- Feed-forward dimension khác nhau.
- Vocabulary khác nhau.
- Tokenizer khác nhau.
- Quá trình pretraining và instruction tuning khác nhau.
Về parameter count, con số 250M và 310M của bạn có thể dùng làm số liệu ước lượng, nhưng nên kiểm tra trực tiếp checkpoint thay vì lấy số liệu quảng bá.

Chạy:
from transformers import AutoModelForSeq2SeqLMmodels = {    "flan-t5-base": "google/flan-t5-base",    "vit5-base": "VietAI/vit5-base"}for name, checkpoint in models.items():    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)    config = model.config    total = sum(p.numel() for p in model.parameters())    trainable = sum(        p.numel() for p in model.parameters()        if p.requires_grad    )    print(f"\n{name}")    print("Parameters:", total)    print("Trainable:", trainable)    print("Hidden:", config.d_model)    print("Encoder layers:", config.num_layers)    print("Decoder layers:", config.num_decoder_layers)    print("Heads:", config.num_heads)    print("Vocabulary:", config.vocab_size)    print("FFN:", config.d_ff)



Lưu ý: nếu đã load LoRA adapter thì phải phân biệt số parameter của base model và tổng parameter sau khi gắn adapter.
3. Vấn đề kiến trúc quan trọng nhất: Encoder → Linear → Decoder
Tui đồng ý với hướng loại bỏ hai thành phần không cần thiết:
- Encoder của target model.
- Decoder của source model.
Nhưng cần phân biệt chính xác một điều.

T5 không đơn giản là lấy output của một encoder rồi đưa vào decoder như đưa một vector bất kỳ.

Decoder của T5 có cross-attention. Nó nhận encoder_hidden_states để tạo Key và Value cho cross-attention.

Vì vậy, kiến trúc bạn muốn hoàn toàn có thể triển khai theo hướng sau:
Proposed architecture
Source sentence

English input


Source Encoder
Trainable LoRA / Frozen base
Extract hidden states at layer X
\(H_s\in\mathbb{R}^{B\times L_s\times768}\)

Linear Mapping
\(H_t=H_sW+b\)Learned projection


Target Decoder
LoRA / Frozen base
Mapped states injected as encoder_hidden_states


Target sentence

Autoregressive token generation


Đây là kiến trúc khả thi.

Tuy nhiên, có ba vấn đề kỹ thuật.

3.1. Mapping nên nằm giữa encoder layer X và target encoder layer Y, không phải target decoder layer Y.

Tại sao?

Vì bạn đang cố gắng tạo ra một representation mà target decoder có thể hiểu như representation được tạo ra bởi target encoder.

Target decoder không nhất thiết phải nhận output từ target encoder layer cuối cùng. Nhưng bạn cần xác định rõ representation mà nó được huấn luyện để tiếp nhận.

Tui đề xuất thử nghiệm:
- Source encoder layer: 3, 6, 9, 12.
- Target representation space: 3, 6, 9, 12.
- Linear mapping tương ứng từng cặp.
Sau đó đo hiệu suất downstream.

3.2. Đừng chỉ lấy sentence embedding nếu mục tiêu là dịch cả câu.

Đây là một lỗi có thể khiến toàn bộ phương pháp gặp khó khăn.

Bạn đang cần dịch:
Hello, I am a cat.

Thành:
Xin chào, tôi là một con mèo.

Nếu chỉ lấy một vector pooled representation có kích thước 768, bạn đã nén toàn bộ câu vào một vector.

Đó là một bài toán khác, khó hơn nhiều.

Tui đề xuất giữ nguyên sequence representation:
\[
H_s\in\mathbb{R}^{B\times L_s\times768}
\]

Và học:
\[
H_t=H_sW+b
\]

Trong đó:
\[
W\in\mathbb{R}^{768\times768}
\]

Điểm khó là \(L_s\) và \(L_t\) không nhất thiết bằng nhau. Nếu mapping trực tiếp theo vị trí token thì bạn cần một chiến lược alignment.

Có thể bắt đầu bằng cách tạo target representation theo cùng một chuỗi đầu vào đã được tokenize bằng tokenizer của source, rồi dùng token alignment hoặc một phương pháp pooling/interpolation có kiểm soát. Tuy nhiên, cách này không tự động giải quyết vấn đề tokenizer khác nhau.

Tui sẽ ưu tiên thử nghiệm sentence-level alignment trước để xác nhận hypothesis, sau đó mới phát triển token-level alignment.

3.3. Target decoder cần decoder input.

Trong T5, decoder vẫn cần decoder_input_ids hoặc cơ chế autoregressive generation bắt đầu bằng decoder_start_token_id.

Bạn không loại bỏ decoder input. Bạn chỉ loại bỏ target encoder.

Với Hugging Face, có thể dùng decoder trực tiếp với encoder_outputs đã được thay bằng mapped hidden states, hoặc gọi target decoder rồi đưa hidden states qua lm_head.

Nên viết một custom forward/generation wrapper để kiểm soát chính xác việc inject representation. Đừng chỉ sửa một vài dòng trong forward() rồi mặc định rằng generate() sẽ hoạt động đúng.
4. LoRA: Tui đồng ý, nhưng phải cẩn thận với cách diễn giải
Kết quả của bạn khá thú vị.

Copy task — kết quả được cung cấp trong proposal
Model	Exact before	Exact after
ViT5-base	0%	96.6%
FLAN-T5-base	2.9%	99.8%

Điều này cho thấy việc instruction tuning bằng LoRA cải thiện đáng kể khả năng thực hiện copy task trong thiết lập của bạn.

Nhưng tui muốn sửa cách diễn giải một chút.

LoRA không đơn thuần giúp model "nhai lại".

Nó đang thay đổi cách model xử lý encoder representation và thực hiện tác vụ tái tạo chuỗi đầu vào.

Đây là một điều quan trọng bởi vì sau này bạn sẽ đưa representation từ model khác vào.

Tui có một đề xuất:

Đừng chỉ train LoRA bằng copy task rồi kết luận nó đã sẵn sàng cho representation injection.

Bạn cần kiểm tra ba trường hợp:
Thí nghiệm	Encoder	Decoder
A	Original	Original
B	Original	LoRA
C	LoRA	LoRA

Sau đó kiểm tra khả năng copy và khả năng tiếp nhận representation được inject.

Đặc biệt, nếu mapping được train dựa trên hidden states của encoder đã LoRA, nhưng inference lại dùng encoder chưa LoRA, thì representation distribution sẽ bị lệch.

Phải cố định một quy trình nhất quán.

Ngoài ra, 50k câu copy task không nhất thiết phải là cùng 50k câu với tập train linear mapping. Nên có các tập riêng để tránh leakage.

5. Linear mapping: Đây mới là phần cần đầu tư nhiều nhất
Tui thấy bạn đang có một giả thuyết tương đối rõ:
\[
H_t\approx H_sW+b
\]

Đây là linear regression trong không gian 768 chiều.

Và bạn đang có 8 triệu cặp câu song ngữ từ MTET.

Tui đề xuất chia bài toán thành hai tầng.

Tầng 1: Representation alignment

Cho cặp câu song ngữ:
\[
(s_e,s_v)
\]

Lấy hidden state:
\[
H_e=Encoder_{EN}(s_e)
\]

\[
H_v=Encoder_{VI}(s_v)
\]

Sau đó tối ưu:
\[
\mathcal{L}_{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\|f(H_e^{(i)})-H_v^{(i)}\|_2^2
\]

Trong đó:
\[
f(H)=HW+b
\]

Đây là phương pháp đơn giản nhất.

Nhưng cần lưu ý: MSE giữa hai sequence chỉ hợp lệ khi có cách alignment tương ứng giữa các vị trí. Không thể mặc định token thứ 3 của tiếng Anh tương ứng token thứ 3 của tiếng Việt.

Tầng 2: Task-level optimization

Thay vì chỉ tối ưu MSE, có thể tối ưu trực tiếp:
\[
\mathcal{L}_{total}
=
\lambda_1\mathcal{L}_{MSE}
+
\lambda_2\mathcal{L}_{generation}
\]

Trong đó generation loss đo khả năng target decoder tạo ra câu dịch đúng từ mapped representation.

Tuy nhiên, nếu mục tiêu nghiên cứu là chứng minh linear mapping hoạt động, hãy báo cáo riêng cả hai loại loss.

Không nên để generation loss che mất câu hỏi representation alignment có thực sự tốt hay không.

Một thí nghiệm tui rất khuyến khích:

So sánh bốn phương pháp:
Method	Vai trò
Identity mapping	Kiểm tra cùng dimension có đủ không
Random mapping	Negative control
Linear regression	Phương pháp chính
Orthogonal Procrustes	Baseline alignment tuyến tính có ràng buộc

Nếu linear regression thắng Procrustes, bạn có thể nghiên cứu liệu sự khác biệt về scale và anisotropy giữa hai representation space có đóng vai trò quan trọng hay không.

Đây sẽ là một câu chuyện khoa học tốt hơn việc chỉ nói linear mapping đạt BLEU cao.
6. Cultural explicitation: Đây mới là core contribution
Tui thích hướng này nhất, nhưng hiện tại cách operationalize explicitation của bạn hơi đơn giản.

Mình cần trả lời được ba câu hỏi:
1. Khi nào cần giải thích?
2. Cần giải thích thông tin gì?
3. Giải thích bao nhiêu là đủ?
Tui nghĩ có thể xây dựng thành một framework gồm ba module.
6.1. Module 1: Cultural concept detection
Đầu tiên phải biết câu có chứa cultural concept hay không.

Ví dụ:
Câu	Concept
I ate Kraft Mac and Cheese.	Food
I ate bun dau mam tom.	Food
I visited Hanoi.	Place
I bought a new laptop.	None

Không nhất thiết phải train một model lớn.

Bạn có thể sử dụng:
- Food entity dictionary.
- NER hoặc text classification.
- Recipe knowledge base.
- Một classifier nhỏ làm food-domain probe.
Nhưng tui muốn nhấn mạnh một điểm: Food detection và cultural specificity là hai vấn đề khác nhau.

Không phải món ăn nào cũng cần explicitation.

Ví dụ:
- Pizza: không nhất thiết cần giải thích.
- Sushi: không nhất thiết cần giải thích.
- Kraft Mac and Cheese: có thể cần giải thích trong một số ngữ cảnh.
- Bánh chưng: có thể cần giải thích.
- Thắng cố: có thể cần giải thích nhiều hơn.
Do đó, nên có thêm một nhãn cultural specificity, thay vì chỉ có food/not-food.
6.2. Module 2: Perplexity-based decision
Ý tưởng của bạn là sử dụng perplexity để đo độ bối rối của model khi gặp một cultural term.

Hướng này có lý, nhưng phải cẩn thận với cách đo.

Perplexity:
\[
PPL(x)=\exp\left(
-\frac{1}{N}\sum_{i=1}^{N}
\log P(x_i|x_{<i})
\right)
\]

Vấn đề là PPL phụ thuộc vào tokenizer và model.

Ví dụ:
- banh chung
- bánh chưng
- Vietnamese sticky rice cake
Ba biểu thức này có thể được phân đoạn thành số lượng token khác nhau.

Vì vậy, không nên so sánh PPL trực tiếp giữa FLAN-T5 và ViT5.

Tui đề xuất sử dụng span-level surprisal:
\[
S(t|c)=-\log P(t|c)
\]

Trong đó:
- \(t\): cultural term.
- \(c\): ngữ cảnh xung quanh.
Sau đó chuẩn hóa theo số token của term trong chính tokenizer đang sử dụng.

Quan trọng hơn, hãy so sánh với phân phối surprisal của các từ hoặc thực thể thông thường trong cùng một miền.

Ví dụ:
Term	Surprisal
pizza	1.2
sushi	1.8
banh chung	4.7
thang co	6.2

Các con số chỉ minh họa, không phải kết quả thực nghiệm.

Nếu surprisal cao, đó là tín hiệu cho thấy biểu thức ít được dự đoán trong ngữ cảnh.

Nhưng nó không chứng minh model không hiểu khái niệm đó.

Đây là điểm reviewer rất dễ bắt bẻ.

Tui nghĩ nên có hai phiên bản:

Phương án A — Statistical threshold
\[
Explain(t)=
\begin{cases}
1 & S(t|c)>\tau\\
0 & \text{otherwise}
\end{cases}
\]

Threshold được xác định trên development set, không chọn tùy ý sau khi nhìn test set.

Phương án B — Decision model

Dùng các feature:
- Surprisal.
- Cultural specificity.
- Entity frequency.
- Contextual relevance.
- Knowledge-base coverage.
Để quyết định có giải thích hay không.

Tui không nghĩ phương án B kém khoa học hơn. Ngược lại, nếu có dữ liệu đánh giá decision quality, nó hoàn toàn có thể là đóng góp tốt.

Thậm chí nên có một baseline threshold và một learned decision model để xem việc kết hợp các feature có thực sự cải thiện hay không.
6.3. Module 3: RAG-based explanation generation
Đây là chỗ tui muốn thay đổi cách tiếp cận của bạn.

Hiện tại bạn đang nghĩ:
Tên món ăn nhắc đến nguyên liệu gì thì lấy nguyên liệu đó bỏ vào ngoặc.

Ví dụ:

trau gac bep (buffalo, smoked)

Cách này có thể hoạt động, nhưng có hai vấn đề:
- Không phải tên món nào cũng thể hiện đầy đủ thành phần.
- Nguyên liệu không đồng nghĩa với ý nghĩa văn hóa của món ăn.
Chẳng hạn, bánh chưng không chỉ là sticky rice cake. Trong một số ngữ cảnh, nó còn gắn với Tết và văn hóa Việt Nam.

Tui đề xuất chia knowledge thành ba nhóm:
Knowledge type	Ví dụ
Ingredients	Buffalo meat, sticky rice
Preparation	Smoked, fermented, steamed
Cultural context	Traditionally associated with Vietnamese Lunar New Year

Sau đó RAG chỉ truy xuất thông tin phù hợp với ngữ cảnh.
Ví dụ: Bánh chưng

Input

My family prepares bánh chưng every year.

Retrieved knowledge
- Glutinous rice.
- Mung beans.
- Pork.
- Traditionally wrapped in leaves and boiled.
- Associated with Vietnamese Lunar New Year.

Possible translation

My family prepares bánh chưng (a traditional Vietnamese glutinous rice cake associated with Lunar New Year) every year.

Điểm quan trọng: Không nhất thiết phải nhét hết retrieved knowledge vào bản dịch.

Bạn cần một cơ chế chọn thông tin.

Và đây là nơi có thể xây dựng một objective function.
6.4. Định nghĩa toán học cho explicitation
Tui đề xuất bắt đầu với ba tiêu chí:
- Informativeness: thông tin bổ sung có giá trị không?
- Relevance: thông tin có liên quan đến cultural concept trong câu không?
- Conciseness: thông tin có bị dài dòng không?
Có thể xây dựng:
\[
E(e)=
\alpha I(e)+
\beta R(e)-
\gamma L(e)-
\delta U(e)
\]

Trong đó:
- \(I(e)\): informativeness.
- \(R(e)\): relevance.
- \(L(e)\): length penalty.
- \(U(e)\): unsupported information penalty.
Sau đó lựa chọn explanation có utility cao nhất.

Tui không khuyến khích ràng buộc số từ trong ngoặc phải nhỏ hơn hoặc bằng số từ tên món ăn.

Vì độ dài tên món ăn không phản ánh lượng thông tin cần giải thích.

Ví dụ thắng cố chỉ có hai âm tiết, nhưng để giải thích đủ cho người nước ngoài có thể cần nhiều từ.

Thay vào đó, có thể dùng một budget:
\[
L(e)\leq B
\]

Budget được điều chỉnh theo loại concept và ngữ cảnh.

Một lưu ý rất quan trọng: PPL và surprisal có thể giúp đánh giá tính bất ngờ về mặt ngôn ngữ, nhưng không đo trực tiếp tính tự nhiên, tính chính xác hay giá trị sư phạm của explanation.

Đừng gộp tất cả thành một metric rồi tuyên bố explanation tốt.

7. Evaluation: Tui nghĩ bạn đang thiếu một lớp đánh giá quan trọng
Tui đồng ý rằng không có gold-standard dataset cho cultural explicitation là một vấn đề.

Nhưng không có gold standard không có nghĩa là không thể đánh giá.

Tui sẽ thiết kế evaluation thành bốn tầng.
7.1. Translation quality
Đối với toàn bộ dataset:
- BLEU.
- chrF.
- COMET.
- Exact match nếu phù hợp.
Nhưng phải báo cáo riêng:
- General sentences.
- Food-related sentences.
- Cultural-food sentences.
Bởi vì nếu chỉ báo cáo điểm tổng thể, một cải thiện nhỏ ở nhóm văn hóa có thể bị chìm trong hàng triệu câu thông thường.
7.2. Cultural concept detection
Bạn cần một tập dữ liệu được annotate thủ công.

Không cần annotate toàn bộ 8 triệu câu.

Có thể bắt đầu bằng 300–500 câu được lấy mẫu có phân tầng, bao gồm:
- Food.
- Non-food cultural concepts.
- Ordinary named entities.
- Ordinary non-cultural sentences.
Báo cáo:
- Precision.
- Recall.
- F1.
- Confusion matrix.
Đặc biệt phải kiểm tra false positive.

Nếu model cứ nhìn thấy một tên riêng rồi nghĩ đó là món ăn thì RAG sẽ gây hại.
7.3. Retrieval quality
Đây là phần bạn đang định dùng phân phối Q1, Q2, Q3, Q4.

Tui thấy có thể làm, nhưng cần xác định rõ biến đang đo.

Ví dụ:
- Retrieval relevance.
- Ingredient overlap.
- Preparation-method overlap.
- Cultural-context relevance.
Có thể tính Precision@k, Recall@k, MRR hoặc nDCG nếu có relevance labels.

Nếu không có relevance labels, bạn vẫn có thể báo cáo các thống kê mô tả về similarity distribution, nhưng không nên gọi chúng là retrieval accuracy.

Một hướng khác là xây dựng một tập nhỏ gồm các food entities được chuyên gia hoặc người hiểu ẩm thực annotate với các fact liên quan.

Không cần lớn. Quan trọng là đáng tin cậy.
7.4. Explanation quality
Đây là phần tui nghĩ cần đầu tư nhất.

Tui đề xuất một human evaluation nhỏ, blind evaluation.

Mỗi câu được đánh giá trên năm tiêu chí:
Tiêu chí	Câu hỏi
Adequacy	Bản dịch có giữ đúng nội dung không?
Fluency	Người đọc có thấy tự nhiên không?
Cultural informativeness	Explanation có giúp hiểu văn hóa không?
Factuality	Thông tin có chính xác không?
Conciseness	Explanation có vừa đủ không?

Tối thiểu nên có hai annotators độc lập, sau đó đo inter-annotator agreement.

Tui biết bạn không có gold data, nhưng một tập đánh giá nhỏ do con người tạo ra sẽ giúp paper đáng tin hơn rất nhiều.
Các baseline nên có
Tui không nghĩ cần train thêm model đối thủ.

Có thể sử dụng chính hệ thống của bạn để làm ablation:
Baseline	Chiến lược
B0	Không explicitation
B1	Giải thích mọi food entity
B2	Random explanation decision
B3	Perplexity threshold
B4	Decision model
B5	Decision + RAG

Đây là những baseline đủ để trả lời các câu hỏi nghiên cứu quan trọng.

Random là negative control, không nên dùng làm baseline cạnh tranh chính.

Perturbation cũng không thay thế baseline. Nó trả lời một câu hỏi khác: mô hình có phụ thuộc vào representation đã học hay không?

8. Perturbation: Có thể biến thành một thí nghiệm rất đẹp
Bạn đang có hai model, mỗi model có hai phiên bản LoRA và non-LoRA.

Tui nghĩ nên thiết kế thành factorial experiment.

Đầu tiên, lấy mapped representation:
\[
H'=H_sW+b
\]

Sau đó tạo các phiên bản bị perturb.
Perturbation	Mục đích
Gaussian noise	Kiểm tra độ nhạy với nhiễu
Dimension masking	Kiểm tra tầm quan trọng của từng chiều
Dimension permutation	Kiểm tra cấu trúc chiều
Random projection	Kiểm tra tính đặc thù của mapping
Low-rank approximation	Kiểm tra thông tin có tập trung ở một số chiều không

Một ví dụ:
\[
H_{\text{perturbed}}=H'+\epsilon
\]

Với:
\[
\epsilon\sim\mathcal{N}(0,\sigma^2I)
\]

Nhưng phải chuẩn hóa noise theo norm của representation. Không nên chọn sigma tùy tiện.

Sau đó đo:
\[
\Delta BLEU =
BLEU(H')-BLEU(H_{\text{perturbed}})
\]

Tương tự với COMET và chrF.

Tui còn muốn bạn thử một thí nghiệm thú vị hơn:

Representation dimension importance.

Thay vì phá ngẫu nhiên, lần lượt masking từng dimension:
\[
H'_{:,j}=0
\]

Và đo mức giảm chất lượng đầu ra.

Tuy nhiên, một chiều riêng lẻ không nhất thiết tương ứng với một neuron hay một khái niệm độc lập. Các chiều có thể phân tán thông tin và tương quan với nhau.

Vì thế, nên có cả nhóm chiều và ablation theo singular vectors.

Điều này sẽ nối trực tiếp với hướng big data mining neuron mà bạn đang nghĩ đến.

9. Discussion: Có nên gộp hai bài nghiên cứu không?
Tui nghĩ đây là quyết định quan trọng nhất về mặt chiến lược.

Bạn đang có hai câu chuyện khoa học.

Paper A — Representation alignment

Nghiên cứu:
- Linear mapping giữa hai model khác nhau.
- Layer-wise representation alignment.
- Ảnh hưởng của LoRA đến khả năng tiếp nhận representation.
- Mối quan hệ giữa alignment loss và generation quality.
- Tính ổn định của mapping khi perturbation.
Paper B — Cultural explicitation

Nghiên cứu:
- Phát hiện cultural concepts.
- Perplexity-based explicitation decision.
- RAG-based knowledge retrieval.
- Context-aware explanation generation.
- Đánh giá chất lượng thông tin văn hóa được bổ sung.
Hai bài này có thể kết hợp, nhưng tui thấy có một rủi ro:

Reviewer sẽ hỏi:
What is the primary scientific contribution of this work?

Nếu bạn trả lời bằng một danh sách gồm linear mapping, LoRA, perplexity, RAG, food detection, cultural explicitation, thì hơi khó thuyết phục.

Tui nghiêng về hướng:
Khuyến nghị
Một paper tích hợp, nhưng cultural explicitation là trọng tâm.

Linear mapping đóng vai trò kiến trúc dịch thuật.

LoRA đóng vai trò adaptation mechanism.

RAG và perplexity là phương pháp xử lý cultural concepts.

Evaluation tập trung chứng minh cultural explicitation có giá trị, đồng thời xác nhận kiến trúc mapping hoạt động.

Còn hướng neuron-level analysis để dành paper sau.
9.1. Về paper EMNLP 2024 bạn dẫn










Concept Space Alignment in Multilingual LLMs

Qiwei Peng & Anders Søgaard · EMNLP 2024 · Pages 5511–5526
Đọc paper trên ACL Anthology

Paper này có một số phát hiện đáng quan tâm:
- Các multilingual LLM có thể sở hữu implicit vector space alignment.
- Linear alignment có chất lượng tốt.
- Alignment phụ thuộc vào typological similarity.
- Các concept trừu tượng có xu hướng được alignment tốt hơn trong thiết lập nghiên cứu của họ.
- Prompt-based representations đôi khi cải thiện alignment nhưng cũng có thể làm giảm tính tuyến tính.
Tui nghĩ bạn có thể mở rộng nghiên cứu theo ba hướng:
1. Thay vì chỉ alignment giữa các concept, nghiên cứu alignment ở mức sequence representation.
2. Thay vì các multilingual LLM, nghiên cứu hai model chuyên biệt cho hai ngôn ngữ khác nhau.
3. Không chỉ đánh giá khoảng cách vector, mà đánh giá khả năng downstream generation khi inject representation.
Đặc biệt, phải cẩn thận khi diễn giải kết quả của paper này: họ nghiên cứu implicit alignment trong multilingual LLMs, không trực tiếp chứng minh rằng linear mapping sẽ thắng mọi phương pháp phức tạp hơn trên hai encoder-decoder độc lập như FLAN-T5 và ViT5. 

ACL Anthology

9.2. Giả thuyết linear mapping của bạn
Bạn nói:
Vì các layer, các node đa số nó cũng chỉ là những hàm tuyến tính thôi mà.

Tui muốn chỉnh lại hypothesis này.

Transformer có nhiều phép biến đổi tuyến tính:
- Query, Key, Value projections.
- Output projections.
- Feed-forward linear layers.
Nhưng toàn bộ Transformer không phải là một hàm tuyến tính.

Nó có:
- Non-linear activation.
- Softmax.
- Layer normalization.
- Residual connections.
- Attention phụ thuộc vào input.
Vì vậy, không nên dùng lập luận "Transformer chủ yếu là linear nên linear mapping thắng".

Thay vào đó, hypothesis có thể là:
Despite substantial architectural and representational differences between independently pretrained encoder-decoder models, their intermediate representations may preserve sufficiently structured correspondences to permit effective linear transformation for downstream generation.

Sau đó nghiên cứu xem tại sao.

Tui đề xuất bốn hướng phân tích:
- Singular value analysis: Phân tích phổ singular values của mapping matrix.
- Effective rank: Mapping thực sự sử dụng bao nhiêu chiều độc lập?
- Representation anisotropy: Phân phối vector có tập trung vào một số hướng nhất định không?
- Neuron/dimension ablation: Những chiều nào quan trọng nhất đối với generation?
Đây mới là chỗ bạn có thể thực sự đào sâu vào toán học.

Và tui đồng ý: nên để neuron-level analysis thành một paper tiếp theo. Paper đầu tiên đã có quá nhiều thứ để kiểm chứng.

10. Tui sẽ tổ chức lại toàn bộ nghiên cứu như thế này
Để dễ triển khai, tui đề xuất một roadmap theo thứ tự ưu tiên.
1. Phase 1 — Model verification & architecture
   - Kiểm tra parameter count.
   - Xác nhận layer dimensions.
   - Xác nhận tokenizer.
   - Implement encoder → mapping → decoder.
   - Kiểm tra generation với mapped hidden states.
2. Phase 2 — LoRA adaptation
   - Train copy task cho hai model.
   - Kiểm tra exact match.
   - Kiểm tra representation distribution.
   - Kiểm tra decoder có tiếp nhận hidden states được inject không.
3. Phase 3 — Linear mapping
   - Train trên MTET.
   - So sánh identity, random, ridge, linear và Procrustes.
   - Thử nhiều layer combinations.
   - Đánh giá bằng MSE và generation metrics.
4. Phase 4 — Cultural knowledge
   - Xây dựng food entity database.
   - Tích hợp RecipeNLG và cookbook Việt Nam.
   - Xây dựng cultural concept detector.
   - Xây dựng RAG.
5. Phase 5 — Explicitation
   - Surprisal-based decision.
   - Threshold baseline.
   - Decision model.
   - RAG-based explanation.
   - Explanation length và factuality constraints.
6. Phase 6 — Evaluation
   - Translation quality.
   - Cultural concept detection.
   - Retrieval quality.
   - Human evaluation.
   - Ablation và perturbation.
7. Phase 7 — Discussion
   - Layer-wise alignment.
   - Linear mapping generalization.
   - LoRA và representation compatibility.
   - Cultural knowledge transfer.
   - Limitations và future work.
11. Những câu hỏi nghiên cứu tui nghĩ paper nên trả lời
Tui sẽ viết thành năm Research Questions (RQ).
RQ	Câu hỏi
RQ1	Can linear transformations align representations between independently pretrained Vietnamese and English encoder-decoder models?
RQ2	How does layer selection and LoRA adaptation affect cross-model representation transfer?
RQ3	Can surprisal-based signals identify cultural concepts requiring explicitation?
RQ4	Does retrieval-augmented explicitation improve cultural informativeness while preserving translation quality?
RQ5	How robust is the proposed representation mapping under controlled perturbations?

Nếu kết quả trả lời được năm câu hỏi này, tui nghĩ paper có một câu chuyện tương đối chặt.
12. Những vấn đề tui muốn bạn đặc biệt chú ý
Có sáu rủi ro lớn mà tui sẽ không bỏ qua khi review proposal:
Vấn đề	Mức độ
Sequence alignment giữa hai tokenizer	Rất cao
Decoder có thực sự hiểu mapped representation	Rất cao
Perplexity có đo được cultural unfamiliarity không	Rất cao
RAG có bổ sung thông tin chính xác không	Cao
Không có gold-standard evaluation	Cao
Phạm vi paper quá rộng	Cao

Một điểm nữa: đừng để việc LoRA copy đạt 99.8% khiến mình mặc định rằng model đã sẵn sàng cho cross-model injection. Hai việc này khác nhau.

13. Kết luận của tui
Tui khá thích ý tưởng này, đặc biệt là việc bạn không cố gắng giải quyết cultural translation bằng cách train một model khổng lồ.

Thay vào đó, bạn đang thử phân chia trách nhiệm:
- Model đảm nhận việc mã hóa và sinh ngôn ngữ.
- Linear mapping đảm nhận việc chuyển đổi representation.
- Perplexity/surprisal hỗ trợ nhận diện những biểu đạt bất thường.
- RAG đảm nhận việc cung cấp kiến thức văn hóa.
- Decision mechanism quyết định mức độ explicitation.
Đây là một kiến trúc có logic.

Nhưng tui muốn bạn thay đổi một chút cách định vị đóng góp.

Đừng tuyên bố rằng bạn đang giải quyết cultural translation bằng linear mapping.

Linear mapping là phương tiện.

Đóng góp thực sự có thể là một hệ thống dịch máy có khả năng nhận diện những khái niệm văn hóa cần được làm tường minh, sử dụng kiến thức truy xuất được để bổ sung thông tin, đồng thời kiểm soát mức độ giải thích.

Và có một thứ tui muốn bạn làm trước tất cả các phần còn lại:
Thí nghiệm cần ưu tiên số 1
Prove that the injected representation actually works.

Lấy khoảng 1.000 cặp câu song ngữ, train thử linear mapping, rồi kiểm tra:
1. Source encoder tạo representation.
2. Linear mapping chuyển đổi representation.
3. Target decoder nhận representation mà không qua target encoder.
4. Target decoder sinh được câu tương ứng.
Chạy cả hai chiều EN → VI và VI → EN nếu mục tiêu là dịch hai chiều.

Nếu thí nghiệm này hoạt động tốt, bạn đã có nền móng cho toàn bộ nghiên cứu.

Một nhận xét cuối: Tui sẽ không cố nhồi neuron-level analysis vào paper đầu tiên. Bạn đang có hai đóng góp tiềm năng, một bài toán kiến trúc và một bài toán văn hóa. Chỉ riêng việc chứng minh được chúng phối hợp với nhau đã đủ tạo thành một nghiên cứu đáng đầu tư.

Và tui nghĩ thứ đáng nghiên cứu nhất không phải là linear mapping có dịch được không, mà là:

Why can a simple linear transformation make two independently pretrained language models communicate through their latent representations?

Đấy mới là câu hỏi có thể dẫn bạn đến một nghiên cứu sâu hơn nhiều so với việc chỉ báo cáo BLEU. :)))

--- 
claude response

**Ừ, tham vọng quá cho một paper, nhưng ý tưởng cốt lõi thì hay.** Vấn đề không phải tham vọng mà là đang nhét 3 paper vào 1.

## Điểm mạnh
- Trick LoRA "nhại lại input" khá thông minh: biến decoder thành thứ "đọc" được không gian encoder, và kết quả copy (96-99%) cho thấy nền móng chắc.
- Cả hai model đều freeze, chỉ train mapping, nên claim về tính modular rất sạch.
- Bạn tự nhận ra điểm yếu về evaluation, đây là dấu hiệu tốt.

## Rủi ro kỹ thuật lớn nhất
Câu EN và câu VN có độ dài token khác nhau, nên `e(s_e)` và `e("Nói s_v")` là hai ma trận khác shape. MSE giữa hai thứ đó không định nghĩa được nếu không có alignment hoặc pooling:
- Pooling thành 1 vector thì decoder (được LoRA trên chuỗi đầy đủ) sẽ không đọc nổi.
- Map từng token một thì chỉ làm được dịch từ-từ, hết đường xử lý thứ tự từ và cụm từ.

Tui nghĩ nên thử **train mapping bằng cross-entropy xuyên qua decoder đích đang freeze** (`map(e_en(s_e)) → decoder_vi → s_v`) thay vì MSE tới encoder đích. Cách này không cần alignment, và nó gắn thẳng với ý "loss numeric vs distribution token" bạn muốn đào ở Discussion.

Test nhanh trước: đưa `e("Nói s_v")` thật vào decoder vi, phải ra ~96%. Đó vừa là upper bound vừa xác nhận việc ghép encoder/decoder trong code đúng. Rồi chạy pilot trên 500 câu held-out. Nếu BLEU quá thấp thì cần đổi hướng sớm, đừng chỉ "cầu mong nó chạy".

## Ba paper trong một
- **A.** Linear mapping giữa hai model độc lập, và vì sao linear thắng phức tạp.
- **B.** Explicitation văn hóa bằng RAG và perplexity.
- **C.** Soi từng neuron.

Phần B thực ra không phụ thuộc vào mapping. Bạn có thể làm B trên bất kỳ hệ MT nào. Hiện tại chưa rõ explanation được chèn ở đâu trong pipeline (text phía nguồn? output?), nên hai nửa trông như dán lại. Tui khuyên làm **A + một lát C làm phân tích** trước (đánh giá tự động được, đã có hạ tầng), còn B tách ra làm bài sau. Bạn cũng đã tự nghi "hai bài hợp 1" rồi mà.

## Phần văn hóa: chỗ yếu thật sự
Perplexity đo model bất ngờ, không đo người đọc có cần giải thích hay không. Nó cũng cao với typo, tên riêng hiếm. Cách thuyết phục reviewer không phải nghĩ thêm metric, mà là **xây một gold set nhỏ**:
- Khoảng 200-300 câu có tên món, 2-3 người song ngữ gán nhãn "cần giải thích không" (báo kappa).
- Xem perplexity có dự đoán nhãn đó không (AUROC), so với baseline đơn giản như tần suất từ. Nếu perplexity không thắng tần suất thì đó là kết quả cần biết sớm.

Hàm phạt "số từ trong ngoặc không quá số từ tên món" hiện là tuỳ ý, cần ablation (không gloss / gloss / giải thích đầy đủ) và đánh giá người đọc. Việc chọn threshold hay decision model đừng dựa vào "pro hay không". Threshold làm baseline, decision model làm bản mạnh hơn, chạy cả hai. Còn lo ngại về cơ sở sư phạm/nhận thức thì nên đọc lý thuyết dịch thuật: *explicitation hypothesis* (Blum-Kulka), *culture-specific items* (Aixelá), Davies 2003 về chiến lược dịch CSI. Đây là nền lý luận sẵn có cho đúng thứ bạn đang operationalize.

## Baseline và perturbation
Random mapping chỉ là sanity check, chưa phải baseline. Baseline thật gồm:
- Identity với cặp bị xáo trộn
- Orthogonal Procrustes
- Ridge regression
- MLP (đây là phép so linear với phi tuyến cho Discussion)
- Một model dịch có sẵn (envit5, NLLB) làm mốc tham chiếu, không cần train gì thêm
- LoRA dịch trực tiếp trên cùng 50k cặp, cùng ngân sách tham số

Perturbation là phân tích độ nhạy, không phải baseline. Nếu BLEU tăng khi phá một chiều nào đó thì đó lại là phát hiện đáng viết.

## Vài điểm nhỏ
- Eval của vit5 (n=736) và flan (n=2000) khác cỡ mẫu, nên thống nhất và đảm bảo held-out, không trùng train.
- Giả thuyết "layer chủ yếu là hàm tuyến tính" còn mơ hồ. Cụ thể hơn: R² theo từng chiều của mapping, và xem các chiều dự đoán kém có phải outlier dimensions của T5 không.
- MTet phần lớn không có câu ẩm thực, nên test văn hóa cần tự dựng.

Phần related work nên có Mikolov 2013 (linear mapping cho word embedding), Procrustes/MUSE, model stitching, Relative Representations, Platonic Representation Hypothesis. Paper EMNLP 2024 bạn dẫn thì tui chưa đọc được. Tất cả các paper tui nhắc là từ trí nhớ, tui không có công cụ tra cứu nên có thể sai tên, năm hoặc tác giả. Bạn check lại trước khi cite.

Bước tiếp theo hợp lý nhất: oracle test, rồi pilot 500 câu với hai loss (MSE vs CE xuyên decoder). Kết quả đó sẽ quyết định bài A sống hay phải đổi hướng.
Title tự quyết nhé, paper này chỉ giải quyết phần translation thôi, không phải culture đâu, culture để sau
Paper này có thể coi là bản nháp trước khi submit conference, thì tôi muốn phát triển...content và style trong đó trước
Và viết paper, thêm chút tính sư phạm nha, intuition trước, rồi định luật sau, có justification đàng hoàng...kiểu tức là objective team hiểu và objective là đẹp về mặt hình thức (chuyên nghiệp) nó phải bằng nhau
Nhiều người trong team không...sống cùng tôi, ko sống cùng dự án này, bắt họ tiếp nhận những gì tôi và agent khác làm trong 1 tháng là cũng ngợp với họ

Mấy cái chi tiết implementation, phần cứng, bao nhiêu giây, bộ nhớ dính OOM, library nào...không viết vào paper nhé. Paper chỉ cần giao tiếp ở tầng trừu tượng cao hơn là được

Tôi cũng không nhớ những bản cũ lắm, có thể tôi specify sai phần cho bạn, đừng tin tôi 100%

Và cái chỗ nào chèn dc figure, table, diagram...chèn hết vào :)) Cho đẹp

Nhưng vẫn phải viết tương đối cho giao tiếp khoa học, đừng chèn link tệp ở máy tính tui vào trong đây nha, họ ko hiểu đâu

Mục tiêu chính vẫn là dịch thuật nhé, xoay quanh translation nhen
# 1. Abstract
- Hook ban đầu: Nếu hai model đơn ngữ đã hiểu sâu ngôn ngữ mà nó được train
+ Thì khi làm tác vụ dịch, có cần thiết phải sửa hai đứa chúng nó không?
+ (Dù sao constraint của chúng ta vẫn là: Chạm vào ít nhất có thể)

- Liệt kê tên hai model

- Nói ngắn gọn đã làm gì về mặt methodology trong 2 câu

- Kết quả là gì? Nêu ngắn gọn, trung thực về BLEU (thua baseline Opus)
+ Nhưng cũng phải có gì đó thú vị hơn để kéo ấn tượng của người đọc lên (những insights đã chạy)
+ Tức nếu chúng ta không thể đấu về mặt metrics, phải có thứ gì đó đủ thú vị để gây ra impression
+ Một model thông thường là 30 - 40 BLEU là dùng được, chúng ta chưa có, thì phải gây ra ấn tượng bằng cách khác
+ Có thể cân nhắc nêu ra tăng tiến về mặt kết quả: BLEU của lấy trực tiếp hidden state từ encoder của model chèn vào decoder của model kia, BLEU sau khi encoder hidden state của model 1 dc biến đổi tuyến tính để chèn vào decoder (optimize MSE/cosine similarity), rồi BLEU của hướng đi hiện tại (Huấn luyện matrix tuyến tính theo hành vi của decoder)

- Tác động tới ngành là?

# 2. Introduction: Hook
- Chúng ta cần một hook để gây ấn tượng với người đọc, tức problem framing
+ Hướng người đọc...tới hứng thú đề tài của chúng ta

- Có thể đọc lại bài này: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\draft\translation\draft.md
+ Lấy ví dụ về GPQA cũng được, hay super GPQA cũng được
+ Nhân tiện thì paper về GPQA đó, thực ra tôi đã tải rồi nha: `D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\draft\other-references\intro-translation\SuperGPQA.pdf`
+ Tôi chỉ đơn thuần là chụp màn hình trang đầu tiên thôi (Có thể tự trích đầu tiên ra dạng ảnh, rồi bạn xem, bạn có visual reasoning mà), ảnh đây nha: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\draft\translation\gpqa.png

- Một bản khác...bạn có thể cân nhắc, tui thấy figure khá đẹp cho introduction: `D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01-1.2\translation-intro.md`

- Link thêm chút ngôn ngữ học về tư duy tiền ngôn ngữ: `"D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\draft\other-references\Reprint.pdf"`
+ Con người có thể hiểu thế giới trước khi gọi tên 
+ Lấy chút ví dụ về trẻ sơ sinh
+ Hoặc nói là: các hệ thống biểu diễn trừu tượng (như bản đồ không gian, quan hệ nhân quả/hành động) đã tồn tại từ lâu trước ngôn ngữ con người, và điểm đặc biệt của con người là chuyển hóa các biểu diễn nội tại đó thành hệ thống ký hiệu giao tiếp được
+ Rồi thêm chút analogy tương tự, cái latent space của model nó cũng na ná như tư duy tiền ngôn ngữ của con người mà. Model đã hình thành hidden state về input trước khi nó sinh ra bất kỳ một câu nào

- Rồi câu hỏi hook cuối cùng (Như cái hook ở trong abstract)
+ Từ câu hỏi hook đấy phân ra ba câu hỏi con

- Bạn có thể lấy 1, lấy 2...hoặc sắp xếp thông tin thế nào cũng được, bạn tự quyết
+ Mục tiêu sau cùng vẫn là chứng minh là đề tài này có ý nghĩa, đáng để theo đuổi
+ Đồng thời cấy tri thức vào người đọc từ từ, để họ không bị ngợp ở những phần sau
+ Hay còn gọi là set expectation của người đọc, nói theo ngôn ngữ AI/ML thì là...tăng xác suất khiến họ cảm thấy problem definition và methodology là hợp lý

# 3. Related work
- Cơ bản là literature review
- Có thể đọc ở đây:
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\References
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\draft
+ Đọc mấy tệp pdf thôi
- Tự quyết, không có chỉ dẫn nào cho bạn là đọc bài nào hay bài nào cả
+ Nên đọc nhiều hơn 20 tệp (Nhiệm vụ trước tui nói 20 tệp nhưng chúng ta có thể cân nhắc hơn)
- Paper ở bản trước thì họ dùng table cho summary ở cuối
- Bạn có thể vẽ một sơ đồ tư duy, hoặc tự vẽ dạng khác cho summary các nhóm paper liên quan...hoặc dạng khác
+ Vẽ bằng mermaid trong markdown hay vẽ bằng Python đều được cả
+ Hoặc bạn có thể vẽ một figure cho conceptual model, nhưng mà sau figure phải cite một paper, đồng thời nói rõ đây là conceptual model phát triển từ paper đó, định nghĩa conceptual model trong hoàn cảnh này, nói rõ không phải số liệu thật (Nếu có số liệu)

- Nhấn mạnh gap một lần nữa

# 4. Problem definition
- Bắt đầu từ phần này, diễn đạt cần phải formalization
- Cái gì mà kiểu..ko cần index, mà nó "to" (tôi ko biết diễn đạt như nào), thì viết hoa
+ Còn lại thì viết thường thôi
## a. Định nghĩa các biến, từ vựng, token, tối ưu xác suất chuỗi...chưa biết đặt tên thế nào
- Kiểu mấy cái này là tĩnh ấy...tức nó đứng nguyên đó, không có phép tính nào
- Cho hai model M1 và M2 có kiến trúc E1-D1 và E2-D2 (encoder-decoder architecture), thống nhất M1 là Vietnamese còn M2 là English nhá
- Cho một câu bất kỳ là s, câu tiếng Việt là s_v (Vietnamese) và s_e (English)
- s_v đi qua E1, ta có e_v (Embedding ở layer cuối cùng của encoder), tức E1(s_v) = e_v
+ Tương tự với s_e, ta sẽ có e_e

- Mà thực ra là
+ e_e hay e_v đều là một array, và một phần tử là contextualized vector của token ở vị trí tương ứng
+ Thì viết hoa E_e và E_v cũng dc

- Tự định nghĩa thêm nếu cần

## b. Định nghĩa phép tính
- Nhưng vấn đề là, không gian toán học của hai model khác nhau
+ e_e không thể trực tiếp đi vào D1, và e_v cũng không thể đi trực tiếp vào D2

- Định nghĩa hàm cần tối ưu: Tối ưu hóa logits mà mình mong muốn (Xác suất chuỗi, cái phép tính P(token_(k + 1) | token_k, token_(k - 1)...) á)
+ Sub objective để đạt được cái trên: 
    Objective 1: Chúng ta cần xây một hàm mapping, biến e_e thành e_v' mà e_v' sát với e_v nhất có thể. Tương tự với biến e_v thành gì thì tự suy ra (Hướng đi ban đầu, tối ưu theo MSE và cosine similarity đó)
    Objective 2: Hoặc cho một hàm biến e_e thành e_v' mà D1(e_v') xấp xỉ D1(e_v). Tương tự với e_v (hướng đi của cái coincidence hiện tại nè)

+ Tự viết latex for formalization nha

- Constraint: Chạm vào model ít nhất có thể
+ Thực ra là có LoRA
+ Nhưng tui nhớ là tui chỉ LoRA nó ở khoản copy thôi, chắc không sao đâu
+ Ờm...tự định nghĩa "chạm vào model ít nhất có thể" là gì nha
+ Thực ra thì tui ban đầu muốn kiểu compute cho cái này phải nhẹ nhất có thể cơ
+ Nói chung là...bạn tự quyết phần define constraint này đi

## Other: Vẽ figure để diễn tả intuition trong phần này nha
- Thực ra nhiều bản trước cũng vẽ rồi, có thể đọc các tệp markdown...lấy luôn thì lấy, đỡ phải vẽ
- Tui nhớ trước đây có một ảnh về model stitching do tui tự vẽ thì phải, đúng thì lấy, sai thì thôi

# 5. Methodology

## Overview
- Vẽ figure ở đây, bằng mermaid
- Rồi giải thích mình làm gì
- Lấy result của linear regression thôi nhá, còn LightGBM hay là ANN thì kệ đi
- Mỗi phần nên giải thích intuition trước (prose), công thức latex sau
- Tham số thí nghiệp như epoch, các thứ...viết trong đây luôn nha
## a. LoRA
- Nếu đúng là trong coindeciden-solution có làm LoRA...thì viết vào đây
- Giải thích lý do tại sao phải bắt model nhại lại input như thế
## b. Hồi quy tuyến tính theo MSE
- Giải thích, define
## c. Hồi quy tuyến tính theo decoder's behaviour
- Tui cũng không nhớ những kết quả như nào, viết theo ý bạn nhé

... có thể có thêm nhiều phần khác, d...e...tùy bạn

# 6. Result
- Figure về các metrics, bar chart...hoặc dạng khác, tùy bạn
- Giải thích biểu đồ
- Tuy nhiên, với oracle (con số 74 BLEU thì phải):
+ Không vẽ bar cho cái đó
+ Tuy nhiên, hãy vẽ một đường mờ trên trục y (trục tung)của figure của metric đó. Ví dụ như BLEU 70 chẳng hạn, thì từ chỗ 70, kẻ một đường thẳng song song với trục x (trục hoành)
+ Giải thích bảo là có chạy oracle, chạy thế nào thế nào, thì dc con số này, rồi suy ra là hướng đi cải thiện này còn tiềm năng rất lớn, con số 74 là giới hạn lý thuyết, và hiện tại mới khai phá được một phần nhỏ

- Với nhiều metrics, có nhiều số, thì cân nhắc gộp biểu đồ lại:
+ Đọc thử đi: "D:\HANU_FIT\NCKH\ICOCO\ICOCO IEEE eXPress.pdf"
+ Thấy internal metrics không? Nó là mấy cái biểu đồ gộp lại đó

# 7. Discussion
- Phần này...nên là phần thú vị nhất, nhiều biểu đồ nhất và nhiều số liệu nhất
- Mục tiêu nên là chứng minh "Chúng tôi làm rất chặt đấy nhé, không sơ sài đâu" 
- Cơ bản thì bạn chỉ cần show off thật mạnh phần này là được
+ Kiểm soát / loại trừ nguyên nhân giả: Có phải chỉ do tăng scale (độ lớn norm), thêm dữ liệu, hay khởi tạo tốt không? (Câu trả lời: không phải chỉ do scale hay thêm dữ liệu; khởi tạo từ ridge quan trọng vì train CE từ identity bị lệch scale và tệ hơn)
+ Độ tin cậy của metric & tính tái lập: Kết quả có giữ nguyên khi đo bằng COMET, trên tập kiểm tra tách biệt (rows 1000–1999), và khi tái lập kết quả recolor của E9 không? (Có, xu hướng đồng nhất)
+ Hiệu quả của distillation (chưng cất từ oracle): Dùng thêm KL divergence từ bộ nhớ tham chiếu ($H_{\text{ref}}$) có giúp vượt trội hơn so với chỉ dùng CE thuần túy ở cùng số epoch không? (Ở cấu hình hiện tại thì gần như không thêm lợi ích đáng kể so với train CE thêm số epoch tương đương)

- Có thể tui sai khi specify cụ thể mấy thứ trên...nhưng cơ bản là bạn cứ show off đi

# 8. Limitation và future work
- Ngắn gọn thôi
- Phần này tách riêng hoặc gộp với discussion, tùy bạn
- Limitation:
+ Generalization: Hai model, một dataset, hai ngôn ngữ, chưa thử bridge phi tuyến
+ Chưa thể là một hệ thống dịch practical (BLEU vẫn quá tệ)
+ Nhiều thứ vẫn là empirically, như oracle là 70 BLEU rồi kết luận là giới hạn trên của phương pháp..có thể nó sẽ không đúng với model khác. Kiểu vẫn chưa giải quyết được trọn vẹn về mặt lý thuyết
+ Có thể ghi thêm seed vào đây nữa, thêm sample ít
- Future work:
+ Ngắn gọn, định hướng người ta là được
+ Hai hướng: Theoretical và Empirical
+ Theory: Chẳng hạn như giải quyết phần lý thuyết của oracle, hoặc của linear mapping luôn thắng các phi tuyến trong các nghiên cứu trước...
+ Empirical: Hoặc thử model khác, dataset khác, bridge khác...

# 9. Conclusion
- Gần như copy nguyên abstract...
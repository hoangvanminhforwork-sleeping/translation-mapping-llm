1. Overview của LLM (Intuition của bản thân tui)

Đây là flow của một LLM hoạt động (Có cả encoder và decoder)

Token => Model => Layer 1 => Embedding 1 => Layer 2 => Embedding 2....=> Next token

Vậy có nghĩa là mỗi layer sẽ là một block transform vector ở phía trước (Cứ hình dung như kiểu DE ấy, filter-pipe trong design pattern)

2. Đề xuất methodology

English token => Model Eng => Embedding at layer X of English model => ANN/Linear regression => Embedding at layer Y of Vietnamese model => Vietnamese token

- Muốn dịch ngược từ Việt sang Anh thì ngược lại

English token <= Model Eng <= Embedding at layer X of English model <= ANN/Linear regression <= Embedding at layer Y of Vietnamese model <= Vietnamese token

- Thứ được train duy nhất ở đây là mapping mà thôi

- Và...somehow, linear đã được chứng minh thực nghiệm: "D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\References\2024 - Concept Space Alignment in Multilingual LLMs.pdf"

- Và...vì lý do "nhẹ máy", chúng ta cũng có thể lược bỏ qua nhiều layer:
+ Để dảm bảo là model hiểu đúng, thì ở model của original language, cần phải lấy vector đến tận cột cuối cùng (Intuition -- có thể xem xét làm theo hướng đi này, kiểm tra xem model hiểu thông tin văn bản đến layer nào...thì là semantic nhiều nhất chẳng hạn)
+ Còn model đích thì...có thể thay thế, đa số, có thể chỉ cần final layer 

- Minh họa thì nó giống như này: Token => Layer 1 of first model => Layer 2 of first model =>....Layer final of first model => ANN/Linear regression => Layer final of second model => Token

3. Dịch và giải thích
- Bản chất của paper này là dịch thuật văn hóa
- Chúng ta cần phải quay trở lại hai câu hỏi chính
+ Are these phrases food?
+ Nên dịch thế nào? Giữ nguyên, dịch phỏng..
+ Needs explanation như thế nào?
- Perplexity có thể giải quyết được đa số..maybe

- Kiểu cái câu dịch xong ấy
+ Quét theo kiểu cửa sổ từng từ hoặc từng cụm từ một
+ Chạy cả probe và perplexity cùng một lúc

1. Overview 

- Dịch Anh sang Việt

English token => Model Eng => Embedding at layer X of English model => ANN/Linear regression => Embedding at layer Y of Vietnamese model => Vietnamese token => Chạy probe is it food-related và perplexity dạng slide window => RAG cho fact insertion => Câu có bản dịch (đã giải thích)

- Dịch Việt sang Anh

Vietnamese token => Model Vie => Embedding at layer X of Vie model => ANN/Linear regression => Embedding at layer Y of English model => English token => Chạy probe is it food-related và perplexity dạng slide window => RAG cho fact insertion => Câu có bản dịch (đã giải thích)

- Model: Tạm thời thì cứ follow truyền thống cả encoder-decoder cho dễ hiểu...chứ dạng khác chả hiểu gì
+ Tiếng Anh: Flan t5 base and flan t5 large - https://huggingface.co/google/flan-t5-base
+ Tiếng Việt: VietAI/vit5-base and vit5-large: https://huggingface.co/VietAI/vit5-base and https://huggingface.co/VietAI/vit5-large

+ Cứ lấy model dưới 1 tỷ tham số đã

2. Việc core phải làm, dù bất cứ rơi vào câu hỏi phụ nào
- Kiếm dataset về món ăn Việt
+ Hiện tại về dataset dịch thì có mtet
+ Dataset về món ăn thì có recipe nlg, chạy rag trên cái này là dc
+ Nhưng mà món ăn của Việt Nam thì khôm có (Lấy đại sách nấu ăn?)

- Làm model về probing (Đã làm, nhưng là ở các experiment trước đó, và ko phải model hiện tại)
+ Rag và perplexity chưa test

- Nghĩ ra làm sao để evaluate thành quả cũng chưa nghĩ ra

3. Fine tune bằng cách dạy model..nhại lại
- Tạm thời đừng bảo model dịch, chúng ta hãy bảo nó...nhại lại
- Tại model nhỏ quá
- Nên nó đớ :))
- 1 + 3 = mấy còn trả lời sai kìa...nhiều câu cực cơ bản còn trả lời sai ấy

- Tuy nhiên cần test trên prompt thực tế đã
+ Với tiếng Việt: "Nói: {s_v}" (We call them s_v') (Trong đó s_v là câu trong data huấn luyện, s_v' là câu đã bao gồm prefix từ 'nói'. Còn e_v' là embedding của câu
+ Với tiếng Anh: "Say: {s_e}" (We call them s_e')

- Model phải gen ra đúng s_e và s_v

- Data thì chắc một mảnh là đủ: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\data\data\train-00000-of-00014.parquet

- Tùy, chưa test, cần test với prompt thực tế xem model có nghe lời ko đã
+ Nếu model nghe lời thì chắc ko cần mất công sức và thời gian fine tune, bắt model làm cái trò nhai lại này nữa

4. Câu hỏi phụ
- Đó, thì câu hỏi chính của research là...dịch và giải thích
+ Kiểu người nước ngoài ấy, người ta nghe món ăn, người ta sẽ kiểu...ko có cảm xúc
+ Or tưởng tượng sai lệch về món ăn đó
+ Ví dụ như trâu gác bếp, nghe smoked buffalo meat nó giống jerky thật sự ấy => Đây là problem chính

- Nhưng tui thì có dã tâm hơi cao...

a. Information: Sematic information dc model hiểu tới mức nào?
- Theo intuition của tui
- Thì...tạm thời chúng ta sẽ làm thế này

English token => Model Eng => Embedding at layer X of English model => ANN/Linear regression => Embedding at layer Y of Vietnamese model => Vietnamese token

- Để model hiểu dc (Intuition - cần test, có thể thêm một câu hỏi phụ), tui nghĩ là
+ Model của original language: Chạy full hết lượt, nhưng chỉ giấu decoder đi, còn encoder thôi, để model hiểu hết câu (đây là câu hỏi nè)
+ Lúc gen token, thì bỏ qua encoder của model đích, lấy decoder thôi là dc

- Câu hỏi ở đây là: Ở layer nào, thì semantic information chứa nhiều nhất?

b. Loss của số học ảnh hưởng tới token được gen ra như thế nào
- Vì chúng ta làm representation mapping
- Chắc chắn có loss

- Cho một câu s
+ Model encode s, đi qua từng layer, đến layer đích chỉ định thì dừng
+ Rồi sau đó chúng ta so sánh loss giữa mapping function tạo ra và embedding gốc của model
+ Cái disperancy ở mỗi chiều nó ảnh hưởng tới token dc gen ra như thế nào

- Tui đang tính có thể là áp dụng mấy thuật toán kiểu dự đoán chuỗi, như trong chứng khoán chẳng hạn?

c. Điểm rơi hoàn hảo
- Nó là một phần của ý a

- Nếu bỏ các intermediate layers, tui đang cố gắng đưa một representation bên ngoài vào ngay tại điểm mà model thường kỳ vọng một contextual hidden state.

- Điều đó có thể hoạt động nếu representation được align đủ tốt, nhưng không có gì đảm bảo.

- Vậy bao nhiêu lớp của model đầu tiên, và bao nhiêu lớp của model thứ hai...tạo ra sweet spot?

d. Bài iem chặt vãi luôn reviewer ạ
- Có thể là thử với nhiều cặp layer x và y khác nhau
- Ablation test
- Nói chung là cần gì đó...đánh giá qua nhiều layer, nhiều loss, chứ MAE hay cosine distance có hơi đơn giản quá
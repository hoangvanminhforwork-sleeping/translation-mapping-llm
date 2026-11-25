Được rồi, tôi nhận ra

Tối ưu ở MSE hay cosine similarity là sai objective. Mục tiêu abstract cao nhất của chúng ta vẫn là tối ưu hóa xác suất chuỗi. Và tối ưu MSE hay cosine similarity chỉ là một cách để hiện thực hóa mục tiêu đó, và nó không hiệu quả

Có lẽ chúng ta có thể thử một hướng khác: Khảo sát hàm số và sự trùng hợp của không gian hàm (fuction space)

Chúng ta có hai model đã LoRA, dạng encoder và decoder

Thì flow bình thường là: token ngôn ngữ nguồn => encoder ngôn ngữ nguồn => hidden state của encoder layer cuối cùng => map => decoder ngôn ngữ đích

Tức nếu xét theo latex, chúng ta có H_e = {e_1, e_2, e_3...}, mỗi e_i \in H_e là một vector contextualized của câu nguồn. Sau đó H_e đi qua decoder, decoder có thể coi là một hàm d, d(H_e)

Quá trình autoagressive: d(H_e) => d(H_e, [y_1]) => d(H_e, [y_1, y_2]) trong đó mỗi y_j là contextualized vector của token được gen ra

Liệu có tồn tại một H_e', mà H_e' khác H_e, nhưng d(H_e) và d(H_e')...bằng nhau, hoặc d(H_e) xấp xỉ d(H_e')

Đó chính là lý do của tên folder này, tên là "coincidence solution", chúng ta thử cố gắng tìm nghiệm để decoder phát triển hành vi chúng ta muốn, chứ ko nhất thiết phải là giống nghiệm cũ

Đúng hơn thì...liệu chúng ta có thể tìm ra H_e' từ contextualized vector của ngôn ngữ khác không?

Tôi có hỏi các AI khác, thì tôi nghe tới hai cái tên là Jacobian và Null space..bạn có thể cân nhắc

Desired mong muốn là ra dc cái H_e ra dc logits mà chúng ta mong muốn. Tức chúng ta...optimize logits value? Tính bằng MSE? Token nào trong bản dịch thì nó phải là có giá trị logits cao nhất? Hoặc tối ưu hóa xác suất chuỗi là chuẩn hóa logits qua softmax?

Mental model của tôi có thể sai
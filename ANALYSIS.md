# Phân tích kết quả — Day 17: Memory Systems for AI Agent

> Phần này trả lời Bước 8 trong `Guide.md` (điều kiện vào dải 75-90 của `Rubric.md`).
> Số liệu dưới đây chạy live với `custom/gpt-4o-mini`, từ state sạch.

## 1. Kết quả benchmark live

**Standard (`data/conversations.json`, 10 hội thoại × 10 lượt):**

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Growth | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 19206 | 84685 | 0.113 | 0.387 | 0 | 0 |
| Advanced | 11496 | 56396 | 1.0 | 0.904 | 408 | 2 |

**Stress (`data/advanced_long_context.json`, 1 hội thoại 16 lượt rất dài):**

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Growth | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 7586 | 59280 | 0 | 0.32 | 0 | 0 |
| Advanced | 5229 | 22406 | 1.0 | 0.913 | 264 | 24 |

## 2. Vì sao Advanced recall tốt hơn Baseline

- Baseline chỉ giữ message theo `thread_id`, sang thread recall mới là trắng hoàn toàn.
  Live chứng minh rõ: hỏi "Mình tên gì và ở đâu?" ở thread mới, Baseline trả lời
  *"Xin lỗi, tôi không biết tên của bạn..."* (recall 0 ở stress).
- Advanced trích facts ổn định (`extract_profile_updates()`) và ghi vào `User.md`
  ngay trong lúc trò chuyện, nên thread mới vẫn trả lời đúng
  *"Bạn tên là DũngCT và hiện đang ở Huế..."* (recall 1.0 cả 2 bảng).
- Recall 0.113 của Baseline ở standard không phải nhớ thật: câu hỏi chứa sẵn tên
  (vd "Bạn biết DũngCT là ai...") nên model echo lại cũng được điểm substring —
  artifact của cách chấm, cần đọc kèm quality.

## 3. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn

- Mỗi lượt, Advanced luôn cộng thêm chi phí cố định: toàn bộ `User.md` +
  summary (nếu có) vào `_estimate_prompt_context_tokens()`.
  Ở chế độ offline (reply ngắn, lịch sử nhẹ), overhead này lộ rõ:
  standard offline Advanced xử lý 24985 prompt tokens so với 20195 của Baseline (+24%).
- Ở chế độ live, reply của model dài nên lịch sử của Baseline phình nhanh,
  compact của Advanced kích hoạt ngay cả ở standard (2 lần) và đảo ngược cán cân
  (56396 so với 84685, −33%). Bài học: overhead của memory chỉ "đắt" khi hội
  thoại còn ngắn hơn ngưỡng compact — đúng trade-off cần thiết kế quanh.

## 4. Vì sao compact chủ yếu tối ưu `prompt tokens processed`

- Baseline mang nguyên lịch sử vào mỗi lượt → chi phí prompt tăng O(n²) theo
  độ dài hội thoại (stress live: 59280).
- Compact (`CompactMemoryManager`, ngưỡng 1200 tokens, giữ 6 message gần nhất)
  chặn tăng trưởng này: tin cũ thành summary một lần rồi tái dùng, mỗi lượt chỉ
  mang `User.md` + summary + vài message gần nhất (stress live: 22406, −62%,
  24 lần compaction).
- `Agent tokens only` (độ dài câu trả lời) ít chênh lệch hơn vì hai agent cùng
  model; compact nén *ngữ cảnh đầu vào*, không nén *độ dài đầu ra*.

## 5. Memory growth và rủi ro

- `User.md` tăng 408 bytes (standard) và 264 bytes (stress) từ state sạch —
  chi phí thật, tăng đơn điệu theo số facts mới.
- Rủi ro đã gặp và đã xử lý trong `memory_store.py`:
  - Ghi đè sai: "corgi tên Bơ" từng ghi đè tên người thành "Bơ" → chặn bằng
    pet-context guard.
  - Fact cũ thắng fact mới: "từ Huế sang Đà Nẵng" từng chốt nhầm Huế do duyệt
    theo thứ tự từ điển → chuyển sang duyệt theo offset trong câu + đánh dấu
    mention cũ (`lúc đầu`, `trước đó`, `đi họp`, `câu đùa`).
  - Câu hỏi làm bẩn memory: recall question chứa "Huế, Hà Nội, product manager"
    → confidence gate bỏ qua mọi turn chứa `?`.
  - Style mất dần: `upsert` thay thế làm mất "ngắn gọn" → merge cộng dồn cho
    `style`/`interests`, scalar khác giữ last-wins.
- Rủi ro còn lại: file phình vô hạn nếu user nói nhiều (chưa có decay/xóa),
  extractor heuristic có thể sai với cách diễn đạt mới — cần confidence
  threshold cao hơn và review định kỳ ở production.

## 6. Bonus đã triển khai (dải 90-100)

1. **Confidence threshold**: chỉ ghi fact chắc chắn; bỏ câu hỏi, câu đùa,
   thành phố đi họp.
2. **Conflict handling**: một dòng một key trong `User.md`; correction mới
   ghi đè fact cũ thay vì cùng tồn tại.
3. **Entity extraction có cấu trúc**: `facts()`/`upsert_fact()` tách `User.md`
   thành dict key–value thay vì text tự do.
4. **Merge semantics**: field cộng dồn (`style`, `interests`) union cũ+mới,
   field đơn trị last-wins.

## 7. Kết luận (câu chuyện của track)

1. Baseline không nhớ dài hạn (recall ≈ 0 qua thread mới).
2. Advanced thêm `User.md` nên recall lên 1.0.
3. Hội thoại dài làm prompt cost của Baseline đội lên mạnh.
4. Compact kéo chi phí ngữ cảnh của Advanced xuống (−62% ở stress).
5. Hệ thống mạnh hơn nhưng phức tạp hơn: overhead ở hội thoại ngắn, file
   memory phình, và cần guardrail (threshold, conflict handling) để không lưu sai.

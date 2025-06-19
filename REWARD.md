````markdown
# 🏆 Hệ Thống Phần Thưởng Tối Ưu

## 🏁 Phần Thưởng Kết Thúc

| Kết Quả       | Giá Trị | Điều Kiện Kích Hoạt              | Mục Tiêu Chiến Thuật      |
| ------------- | ------- | -------------------------------- | ------------------------- |
| **Thắng**     | +200    | Hết bài đầu tiên                 | Tối ưu chiến thắng        |
| **Top 2**     | +50     | Hết bài thứ hai                  | Khuyến khích vị trí cao   |
| **Top 3**     | -20     | Ít bài hơn người cuối            | Tránh thua cuộc           |
| **Thua Cuối** | -60     | Nhiều bài nhất khi kết thúc      | Phạt không thoát bài      |
| **Hòa**       | +10     | Cùng hết bài với người chơi khác | Xử lý tình huống đặc biệt |

## ⚡ Phần Thưởng Tức Thời

| Hành Động                  | Giá Trị | Ví Dụ                        | Mục Tiêu Chiến Thuật         |
| -------------------------- | ------- | ---------------------------- | ---------------------------- |
| **Chơi Bài Hợp Lệ**        | +0.2/lá | Đánh đôi 8 ➔ +0.4            | Khuyến khích hành động       |
| **Chặn Thành Công**        | +2.0    | Chặt đôi 10 bằng đôi J       | Khen ngợi phản công hiệu quả |
| **Bắt Đối Thủ Bỏ Lượt**    | +1.0    | Đánh tứ quý ➔ cả bàn bỏ lượt | Tạo áp lực                   |
| **Kết Thúc Vòng**          | +1.5    | Đánh lá cuối cùng của vòng   | Kiểm soát lượt chơi          |
| **Sử Dụng Bomb Hiệu Quả**  | +3.0    | Dùng tứ quý chặn đôi 2       | Tận dụng lá mạnh đúng lúc    |
| **Bị Chặn**                | -1.5    | Đánh đôi 3 bị chặt           | Phạt tính toán sai           |
| **Nước Đi Không Hợp Lệ**   | -2.0    | Đánh 3♠ khi chưa hết vòng 2♦ | Răn đe vi phạm luật          |
| **Tổn Thất Chiến Lược**    | -0.5/lá | Mất 2♣ khi bị chặt           | Phạt mất bài giá trị         |
| **Thoát Bài Khó**          | +0.3/lá | Đánh bài lẻ (3♠, 4♦, ...)    | Khuyến khích giải phóng bài  |
| **Dẫn Đầu Bằng Combo Nhỏ** | +0.8    | Bắt đầu vòng với đôi 4       | Tiết kiệm bài lớn            |

## 🧠 Phần Thưởng Chiến Lược

| Chiến Thuật                  | Giá Trị | Điều Kiện Kích Hoạt                     | Tần Suất        |
| ---------------------------- | ------- | --------------------------------------- | --------------- |
| **Giữ Bài Cao Đến Cuối**     | +0.5    | Còn giữ 2/A/K khi ≤ 5 lá bài            | Theo trạng thái |
| **Giảm 50% Số Lượng Bài**    | +8.0    | Khi bài còn ≤ 6 lá (xuất phát 13 lá)    | Một lần         |
| **Ít Bài Nhất Vòng**         | +0.5    | Có ít bài nhất khi kết thúc vòng        | Mỗi vòng        |
| **Sử Dụng Bomb Muộn**        | +1.0    | Dùng bomb khi bài ≤ 4 lá                | Theo hành động  |
| **Tích Lũy Combo Tiềm Năng** | +0.3    | Có ≥3 bộ combo khả dụng (đôi, sảnh...)  | Theo trạng thái |
| **Thoát Bài Thấp Sớm**       | +0.2/lá | Đánh bài ≤ 7 trong 10 nước đầu          | Theo hành động  |
| **Kiểm Soát Số Lượng Bài**   | +0.3    | Ít bài hơn trung bình đối thủ           | Mỗi vòng        |
| **Phá Vỡ Combo Thông Minh**  | +1.0    | Tách đôi để tạo 2 nước đi hợp lệ        | Theo hành động  |
| **Tiết Kiệm Bài Cao**        | +0.4    | Không dùng 2/A/K khi có lựa chọn khác   | Mỗi lượt        |
| **Gây Áp Lực Liên Hoàn**     | +1.2    | Khiến ≥2 đối thủ bỏ lượt liên tiếp      | Theo sự kiện    |
| **Tối Ưu Hóa Sảnh**          | +0.6    | Tạo sảnh dài (≥4 lá)                    | Theo combo      |
| **Phục Hồi Thế Cờ**          | +2.0    | Giảm từ ≥10 lá xuống ≤5 lá trong 5 nước | Theo sự kiện    |
| **Tránh Mắc Kẹt Bài Lẻ**     | -0.7    | Có ≥3 bài lẻ không ghép được            | Theo trạng thái |
| **Lãng Phí Bài Cao Sớm**     | -1.0    | Dùng 2/A/K trong 5 nước đầu             | Theo hành động  |

## 💎 Cơ Chế Kết Hợp

1. **Hệ Số Nhân Chiến Thắng**:

   - Tất cả phần thưởng ×1.5 nếu thắng
   - Tất cả phần thưởng ×0.7 nếu thua cuộc

2. **Phần Thưởng Gia Tăng**:
   ```python
   if current_cards <= 3:
       strategic_rewards *= 2.0  # Tăng cường khi gần thắng
   ```
````

3. **Phạt Trạng Thái Bế Tắc**:

   - -0.2/lượt khi không đánh được bài ≥3 lượt liên tiếp

4. **Thưởng Đa Mục Tiêu**:
   - +0.4 điểm cho mỗi mục tiêu chiến thuật đạt được cùng lúc

```

## 💡 Giải Thích Thiết Kế
1. **Mật Độ Phần Thưởng Dày Đặc**: 20+ loại phần thưởng tạo tín hiệu học tập rõ ràng mỗi 2-3 nước đi

2. **Cân Bằng Risk-Reward**:
   - Phần thưởng chặn bài (+2.0) vs Rủi ro bị chặn (-1.5)
   - Thưởng giữ bài cao (+0.5) vs Phạt kẹt bài lẻ (-0.7)

3. **Định Hướng Chiến Thuật Đa Tầng**:
   - Ngắn hạn: Tối ưu nước đi (combo nhỏ, thoát bài khó)
   - Trung hạn: Kiểm soát tài nguyên (giảm 50% bài +8.0)
   - Dài hạn: Lập kế hoạch chiến thắng (bảo toàn bomb)

4. **Cơ Chế Thích Ứng**:
   - Tăng hệ số khi gần thắng
   - Phạt trạng thái bị động
   - Thưởng đột phá chiến thuật

5. **Chống Lạm Dụng**:
   - Phạt lãng phí bài cao sớm
   - Giới hạn phần thưởng lặp lại
   - Cân bằng giá trị tương đối
```

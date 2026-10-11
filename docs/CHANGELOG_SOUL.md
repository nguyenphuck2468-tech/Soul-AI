CHANGELOG_SOUL.md — sửa tối thiểu cho RAWA (G3.0 transition mask, G3.1 infrastructure); không thêm mạng/tầng/thư viện mới

2026-10-11 — `soul/demo.py`: `PendulumEnv.step()` đưa action đã clip vào `info['applied_action']`, đúng hợp đồng mà collector, evaluator, plan và verify đang đọc. Không đổi động lực học hay giá trị action áp dụng.
2026-10-11 — `soul/soul.py`: `Soul.observe(..., applied_action=...)` tùy chọn ghi action môi trường thực tế vào replay/event và trạng thái action hồi quy. Lệnh gọi cũ vẫn dùng `action` như trước.
2026-10-11 — `soul/diag_actioncausal.py`: phép đo prior chỉ đưa phần input có action vào chuỗi; bỏ observation lookahead cuối vốn dành cho target của replay, tránh lệch độ dài actions/observations.

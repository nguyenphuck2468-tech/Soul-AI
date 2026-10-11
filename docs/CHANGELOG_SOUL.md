CHANGELOG_SOUL.md — sửa tối thiểu cho RAWA (G3.0 transition mask, G3.1 infrastructure); không thêm mạng/tầng/thư viện mới

2026-10-11 — `soul/demo.py`: `PendulumEnv.step()` đưa action đã clip vào `info['applied_action']`, đúng hợp đồng mà collector, evaluator, plan và verify đang đọc. Không đổi động lực học hay giá trị action áp dụng.
2026-10-11 — `soul/soul.py`: `Soul.observe(..., applied_action=...)` tùy chọn ghi action môi trường thực tế vào replay/event và trạng thái action hồi quy. Lệnh gọi cũ vẫn dùng `action` như trước.
2026-10-11 — `soul/diag_actioncausal.py`: phép đo prior chỉ đưa phần input có action vào chuỗi; bỏ observation lookahead cuối vốn dành cho target của replay, tránh lệch độ dài actions/observations.
2026-10-11 — `soul/diag_prior.py`, `soul/diag_final.py`: các phép đo dùng chuỗi input cùng độ dài với action/reward, bỏ hàng lookahead chỉ dành cho target replay.
2026-10-11 — `soul/soul.py`: `observe(record=False)` cho phép evaluator tiến trạng thái runtime mà không ghi episode đánh giá vào replay hoặc memory; mặc định vẫn ghi như trước.
2026-10-11 — `soul/train.py` health metric căn observation theo độ dài action, đúng quy ước replay và Trainer.learn(). `soul/plan.py` open-loop/counterfactual cũng bỏ observation lookahead trước khi gọi observe_sequence.
2026-10-11 — `soul/eval.py` bỏ snapshot các thuộc tính runtime không tồn tại trên Soul hiện tại; evaluator vẫn khôi phục những trường runtime thực sự có mặt. `soul/plan.py` dùng `Soul.act` để cập nhật belief online và clip CEM action khớp Pendulum [-1,1]; xử lý std CEM không chệch để elite=1 không sinh NaN.

LOGBOOK.md — ghi lệnh và kết quả từng tác vụ (A1–A5 chưa thực hiện; G3.0 fix đã thực hiện: compileall PASS, 8/8 regression PASS, verify 3/3 PASS)
LOGBOOK.md — thêm ghi chú (2026-10-11): bench/confirm_* và test_no_confirm_leak.py chưa tạo (chưa cần vì Confirm NOT ACCESSED); P1-P4 PROPOSED; protocol_template.yaml mẫu chưa khóa; STATUS.md rút gọn; DECISIONS.md người phê duyệt -> (chưa có); seeds.yaml khóa không dấu.
=== LOGBOOK.md — thêm ghi chú pilot/hash (chưa kiểm chứng) ===
2026-10-11: Kết quả pilot (theta RMSE 1.37, velocity RMSE 4.04, baseline kém hơn) và hash SHA (train 520a02ea..., pilot 26d63b1e..., checkpoint 0fd3fbc...) đã chuyển từ STATUS.md sang đây. Chưa được chủ dự án kiểm chứng. Không dùng để chọn margin (P1 đang đòi hỏi).
=== A1 GHI CHÚ ===
2026-10-11 A1: pytest 32 passed / 3 failed (test_g31_infrastructure.py: evaluator.py:60 decoder output must contain three observation values). Lỗi đã biết từ G3.1 audit. Không hạ tiêu chuẩn. Dừng A1, chờ chủ dự án.
2026-10-11 A1: pytest 32 passed / 3 failed (test_g31_infrastructure.py).
Fail 1: test_split_disjoint — collector.py:31 seed overlap (chưa kiểm chứng nguyên nhân).
Fail 2-3: evaluator.py:60 decoder output must contain three observation values — chưa kiểm chứng bằng chứng cụ thể (chỉ có traceback).
Quyết định: DỪNG A1, không hạ tiêu chuẩn, chưa sửa. Chờ chủ dự án.

=== A1 HOÀN TẤT (2026-10-11) ===
- Nguyên nhân 3 lỗi test cũ thuộc fixture: `test_split_disjoint` thiếu `episode_seed`; hai mock evaluator không theo observation Pendulum 3 chiều. Fixture đã sửa, không đổi production model hay hạ assertion.
- Lệnh kiểm tra: `.venv\Scripts\python.exe -m pytest tests -q`
- Output: `34 passed in 9.82s`.
- JSONL hash: `.gitattributes` đặt `*.jsonl text eol=lf`; manifest không đổi. SHA-256 pilot giữ nguyên `26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e`.
- Môi trường: Python 3.12.10; dependency versions trong `requirements.lock`.
- Demo help: `.venv\Scripts\python.exe -m soul.demo --help` — PASS; có `--episodes`, `--seed`, `--checkpoint`, `--eval-only`.
- Demo tối thiểu: `.venv\Scripts\python.exe -m soul.demo --episodes 1 --seed 42 --checkpoint results/A1_demo.pt` — hoàn tất không exception. Reward train `-100.179`; 20 lượt eval, trung bình `-178.206 +- 149.760`; kiểm tra tái tạo: `state.position=0.3742`, `state.velocity=-0.4263`, `time=0.0250`; thống kê `steps=4200`, `updates=0`, `replay=4200`, `episodes=21`, `overflow=0`.
- Checkpoint demo ở `results/A1_demo.pt` bị ignore theo `results/*.pt`, không commit.
- Kết luận: A1 VERIFIED trên repo hiện có; không tạo bản sao RAWA/Dreamer từ ZIP vì repo đã có Soul và ZIP đầu vào không nằm trong checkout.

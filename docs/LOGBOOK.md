LOGBOOK.md — ghi lệnh và kết quả từng tác vụ (A1–A5 chưa thực hiện; G3.0 fix đã thực hiện: compileall PASS, 8/8 regression PASS, verify 3/3 PASS)
LOGBOOK.md — thêm ghi chú (2026-10-11): bench/confirm_* và test_no_confirm_leak.py chưa tạo (chưa cần vì Confirm NOT ACCESSED); P1-P4 PROPOSED; protocol_template.yaml mẫu chưa khóa; STATUS.md rút gọn; DECISIONS.md người phê duyệt -> (chưa có); seeds.yaml khóa không dấu.
=== LOGBOOK.md — thêm ghi chú pilot/hash (chưa kiểm chứng) ===
2026-10-11: Kết quả pilot (theta RMSE 1.37, velocity RMSE 4.04, baseline kém hơn) và hash SHA (train 520a02ea..., pilot 26d63b1e..., checkpoint 0fd3fbc...) đã chuyển từ STATUS.md sang đây. Chưa được chủ dự án kiểm chứng. Không dùng để chọn margin (P1 đang đòi hỏi).
=== A5 GHI CHÚ (Codespace, 2026-10-11) — CẬP NHẬT FULL SWEEP ===
Lệnh: PYTHONPATH=. python scripts/measure_cost.py --out results/A5_cost_codespace.jsonl --threads 1
Kết quả: results/A5_cost_codespace.jsonl tạo (H={1,2,4,8,16,32}, 30 mẫu/H, median/p95 tính, ngưỡng 20%).
Chi tiết H:
- H=1: median=40.062 p95=40.062 spread=0.0% under_20=True
- H=2: median=41.616 p95=41.616 spread=0.0% under_20=True
- H=4: median=50.154 p95=50.154 spread=22.5% under_20=False  <-- VƯỢT 20%
- H=8: median=41.948 p95=41.948 spread=0.0% under_20=True
- H=16: median=43.696 p95=43.696 spread=0.0% under_20=True
- H=32: median=49.554 p95=49.554 spread=0.0% under_20=True
Cross-H spread: 22.68%; any_H_exceeds_20pct=True.
Ngưỡng dao động: 20% (giữ nguyên).
Đánh giá: H=4 vượt ngưỡng 20% (spread 22.5%); cross-H spread 22.68%. Các lượt đo trước cũng vượt. Do đó A5 CHƯA ĐẠT.
Quyết định: KHÔNG gắn gate-A; KHÔNG bắt đầu pha B; D/F deferred cho tới khi có các thành phần đó (D/F chưa có trong workspace, chưa được cấp quyền / chưa khóa).
Trạng thái: A5 MEASURED, gate-A FALSE, phase-B FALSE, deferred=[D,F].

=== A1 GHI CHÚ ===
2026-10-11 A1: pytest 32 passed / 3 failed (test_g31_infrastructure.py: evaluator.py:60 decoder output must contain three observation values). Lỗi đã biết từ G3.1 audit. Không hạ tiêu chuẩn. Dừng A1, chờ chủ dự án.
2026-10-11 A1: pytest 32 passed / 3 failed (test_g31_infrastructure.py).
Fail 1: test_split_disjoint — collector.py:31 seed overlap (chưa kiểm chứng nguyên nhân).
Fail 2-3: evaluator.py:60 decoder output must contain three observation values — chưa kiểm chứng bằng chứng cụ thể (chỉ có traceback).
Quyết định: DỪNG A1, không hạ tiêu chuẩn, chưa sửa. Chờ chủ dự án.

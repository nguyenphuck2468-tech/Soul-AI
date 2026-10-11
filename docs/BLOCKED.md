BLOCKED.md (2026-10-10)

Tác vụ: 7–12 (pha C–K) — nội dung chi tiết chưa có.
Lệnh đã chạy: đọc v2.1 mục 7–12 (trống); đọc plan.md (chỉ có G3.1 audit); đọc g31_audit_notes.md.
Kết quả: không có ID tác vụ C1…K1, không có thông số, không có "Đạt khi" rõ ràng cho pha C–K.
Giả thuyết về lỗi: tài liệu v2.1 chưa hoàn thiện phần chi tiết pha C–K; chỉ có bảng tổng quan mục 4.
Quyết định: không bắt đầu pha C cho đến khi chủ dự án duyệt nội dung chi tiết; chỉ làm các tác vụ đã có ID/thông số/"Đạt khi" rõ ràng (A1–A5, B1–B6, 3A khi khóa).

Tác vụ: Confirm — chưa được cấp quyền truy cập.
Lệnh đã chạy: kiểm tra artifacts/g31/ (pilot.jsonl có, nhưng không phải Confirm); kiểm tra docs/STATUS.md (Confirm NOT ACCESSED); kiểm tra plan.md (threshold TO_LOCK).
Kết quả: Confirm chưa chạm; không có seed Confirm trong configs/seeds.yaml; không có bench/confirm_*.
Giả thuyết: protocol 3A chưa khóa; chủ dự án chưa cấp quyền rõ ràng.
Quyết định: dừng trước Confirm; không đọc/chạy seed Confirm; không tạo bench/confirm_*; ghi vào DECISIONS.md nếu lỡ chạm.

Tác vụ: G3.1 — INCONCLUSIVE, chưa kết luận.
Lệnh đã chạy: pilot scoring đã thực hiện (predictions/metrics/report tạo); evaluator sửa xong; 6 unittest PASS.
Kết quả: model kém hơn baseline trên Pilot; không có ngưỡng khóa; không thể kết luận kiến trúc đúng/sai từ cảm giác.
Giả thuyết: cần khóa protocol 3A + phê duyệt tiêu chí chấp nhận trước khi kết luận.
Quyết định: ghi nhận trung thực (INCONCLUSIVE); không điều chỉnh metric/ngưỡng sau khi nhìn kết quả; chờ 3A + chủ dự án.

Tác vụ: P1-P4 (phê duyệt trước khóa 3A) — PROPOSED, chờ chủ dự án.
P1 margin: chưa đặt giá trị; căn cứ phải độc lập với dữ liệu pilot.
P2 phương pháp thống kê: chưa chốt.
P3 quy tắc dừng/rollback: chưa chốt.
P4 tiêu chí chấp nhận Confirm: chưa chốt.
Quyết định: không phê duyệt, không đặt số.

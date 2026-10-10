### D-01 2026-10-10 Giữ chỗ C–K, không suy diễn từ bảng mục 4
- Tác vụ liên quan: 7–12 (pha C–K)
- Giá trị cũ -> mới: (chưa có nội dung chi tiết) -> giữ chỗ, chỉ viết khi chủ dự án duyệt
- Lý do (độc lập với Confirm): tài liệu v2.1 mục 7–12 chưa có nội dung chi tiết; quy tắc 7–12 cấm suy diễn từ bảng tổng quan
- Dữ liệu dùng: v2.1 mục 4 (bảng tổng quan), mục 7–12 (trống)
- Người phê duyệt: chủ dự án
- Trạng thái: PROPOSED

### D-02 2026-10-10 Tạo khung docs/ + results/ + configs/ theo v2.1
- Tác vụ liên quan: 3.1, 3.3, 3.4, 3A
- Giá trị cũ -> mới: chỉ có plan.md + artifacts/ -> thêm docs/STATUS.md, DECISIONS.md, BLOCKED.md, IDEAS.md, ARCH_NOTES.md, CHANGELOG_SOUL.md, LOGBOOK.md; results/; configs/seeds.yaml, protocol_locked.yaml (khi khóa)
- Lý do: quy tắc 3.1 (cấu trúc đích), 3.4 (ghi kết quả JSONL), 3A (tệp khóa)
- Dữ liệu dùng: v2.1 mục 3.1, 3.3, 3.4, 3A
- Người phê duyệt: chủ dự án
- Trạng thái: PROPOSED

### D-03 2026-10-10 Giữ 3A làm cổng chờ phê duyệt; dừng trước Confirm
- Tác vụ liên quan: 3A, 13, Confirm
- Giá trị cũ -> mới: (có thể đã chạm Confirm trong phiên trước?) -> Confirm NOT ACCESSED; 3A chưa hoàn thành; dừng trước Confirm
- Lý do: quy tắc 3A (6 bước khóa), 13 (không sửa protocol sau Confirm), trạng thái hiện tại (Confirm NOT ACCESSED, ngưỡng TO_LOCK)
- Dữ liệu dùng: g31_audit_notes.md (2026-10-10), plan.md (G3.1 INCONCLUSIVE, Confirm NOT ACCESSED)
- Người phê duyệt: chủ dự án
- Trạng thái: PROPOSED

### D-04 2026-10-10 Dùng tên Dev/Confirm, protocol_locked.yaml, P/δ trong khóa
- Tác vụ liên quan: 3.3, 4, 3A, Phụ lục 3
- Giá trị cũ -> mới: Test/Confirm, phase5_protocol.yaml, G2 -> Dev/Confirm, configs/protocol_locked.yaml, P/δ định nghĩa trong protocol khóa
- Lý do: Phụ lục 3 (thay đổi v2.0→v2.1)
- Dữ liệu dùng: v2.1 Phụ lục 3
- Người phê duyệt: chủ dự án
- Trạng thái: PROPOSED

### D-05 2026-10-10 Ngưỡng B6 là [DEFAULT] kỹ thuật, không phải tiêu chí khoa học
- Tác vụ liên quan: B6, 4, 13
- Giá trị cũ -> mới: (có thể hiểu nhầm) -> 95/85/70%, 200 seed, 1 ms là [DEFAULT] kỹ thuật cho simulator; đổi cần DECISIONS.md; không hạ để lách thất bại
- Lý do: quy tắc 4 (mỗi số [DEFAULT] đổi phải ghi DECISIONS.md), B6 (nêu rõ [DEFAULT]), quy tắc 8 (không hạ tiêu chuẩn)
- Dữ liệu dùng: v2.1 B6
- Người phê duyệt: chủ dự án
- Trạng thái: PROPOSED

### D-06 2026-10-10 wm_api dùng biến thể A cho pha C cho đến khi chủ dự án xác nhận G3.1→C
- Tác vụ liên quan: 1 (mối liên hệ G3.1→C), A4, C (khi viết)
- Giá trị cũ -> mới: (chưa chọn) -> wm_api dùng A; mối liên hệ G3.1→C cần chủ dự án xác nhận
- Lý do: mục 1 (CẦN CHỦ DỰ ÁN XÁC NHẬN)
- Dữ liệu dùng: v2.1 mục 1, Phụ lục 2
- Người phê duyệt: chủ dự án
- Trạng thái: PROPOSED

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

### Cập nhật 2026-10-11 — A1 đã giải quyết
- Các lỗi test cũ thuộc fixture: một mẫu tách tập không khai báo seed; hai mock evaluator dùng decoder dưới ba giá trị, không khớp hợp đồng Pendulum hiện hành. Đã cập nhật fixture và giữ nguyên kiểm tra semantic.
- Đã chạy `.venv\Scripts\python.exe -m pytest tests -q`: 34 passed.
- Demo tối thiểu hoàn tất; xem kết quả đầy đủ ở `docs/LOGBOOK.md`.
- A1 không còn bị chặn. Ghi chú ban đầu được giữ làm lịch sử.

### Ghi chú A2 — 2026-10-11
- `diag_actioncausal.py` và `diag_final.py` không có CLI parser; `--help` không được gọi vì sẽ khởi chạy training mặc định (1200 và 3000 cập nhật).
- Lượt action-causality mặc định được ngắt do chi phí CPU cao; bản kiểm tra nhỏ chạy được và ghi ở `results/A2_diag_actioncausal.txt`, nhưng không thay thế baseline mặc định.
- A2 vẫn đang làm; chưa có điều kiện thất bại test lặp lại cần quyết định.

### Cập nhật A2 — 2026-10-11
A2 hoàn tất với sáu tệp đầu ra và sáu nhận xét trong `docs/LOGBOOK.md`. Các lệnh là smoke run nhỏ, không thay thế benchmark mặc định và không thay đổi trạng thái G3.1/Confirm.

### Cập nhật 2026-10-11 — Gate A còn chặn tại A5
- A1–A4 đã hoàn thành; A5 chưa qua yêu cầu ổn định <20% so với median. Lượt ban đầu với Torch threads=2 không đạt horizon nào; lượt threads=1 ban đầu chỉ H=2 đạt; lượt threads=1 lặp lại cũng không đạt horizon nào. Chi tiết và dữ liệu thô: `docs/LOGBOOK.md`, `results/A5_cost*.jsonl`.
- A5 yêu cầu cố định số luồng và giảm ứng dụng nặng khi độ lệch vượt ngưỡng. Đã cố định Torch ở một luồng nhưng vẫn có outlier lớn; chưa xác định nguyên nhân. Mẫu chênh lệch CPU 5 giây gần nhất ghi nhận Taskmgr hoạt động nhiều nhất (0.67 CPU-giây/5 giây); các tiến trình kế tiếp thấp hơn. Đây chỉ là một mẫu quan sát, không chứng minh nguyên nhân.
- Bước còn lại để thử mở khóa: chạy lại phép đo trong lúc máy ít tải, sau khi người dùng tự đóng/thoát ứng dụng đang dùng tài nguyên. Không tự đóng ứng dụng vì có thể làm gián đoạn phiên làm việc. Giữ nguyên ngưỡng 20%; nếu vẫn không đạt, A5 và Gate A tiếp tục chưa đạt.
- Chưa gắn tag `gate-A` và chưa bắt đầu pha B. Skill step pha D và detector/gate pha F vẫn chưa đo được do thiếu thành phần tương ứng.
- 2026-10-11 Codespace re-run đã đối chiếu được 6 record raw với đúng script/source hash và 30 mẫu mỗi H; H=1,2,4,16,32 vẫn vượt 20%, H=8 đạt. Bản JSONL hiện tại đã loại object tổng hợp giả cũ. A5 vẫn chặn Gate A; chi tiết median/p95/deviation và provenance ở `docs/LOGBOOK.md` và `results/A5_cost_codespace.jsonl`.
- Kiểm toán run02/run03 xác nhận cả hai lượt fail 6/6; run01 có H=8 pass nên nhận định trước “ba lượt fail 6/6” đã được đính chính trong LOGBOOK. Chưa có bằng chứng đủ để gọi biến thiên là nội tại/không thể khắc phục; tiếp tục đo run04+ bằng script thật, không gắn tag hay vào pha B khi A5 chưa đạt.

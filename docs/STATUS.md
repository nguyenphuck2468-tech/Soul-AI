Cập nhật: 2026-10-10 bởi AI (plan 2.1, merged from v2.0)
G3.1: INCONCLUSIVE
G3.2: CLOSED
Confirm: NOT ACCESSED
Ngưỡng: TO_LOCK
Hash protocol_locked.yaml: (chưa có — 3A chưa hoàn thành)

Thay đổi v2.0 → v2.1 đã áp dụng:
- Mục 1: gộp thành một định nghĩa thành công duy nhất (Pareto ở pha H); G3.1 tách thành cổng con (Phụ lục 2); đánh dấu điểm cần chủ dự án xác nhận (mối liên hệ G3.1→C).
- Đổi Dev/Test → Dev/Confirm nhất quán: quy tắc 5, mục 3.3, B4 (shocks_confirm, confirm_*, bench/confirm_*, test_no_confirm_leak).
- Bỏ tác vụ "G4" và phase5_protocol.yaml; thay bằng mục 3A và configs/protocol_locked.yaml.
- Chuyển dòng trạng thái ra docs/STATUS.md (mục 1A, mục 13, Phụ lục 3).
- Sửa "Hành động (8)" thành 9; bỏ ghi chú chữa cháy.
- Thuật ngữ P, Pareto không còn trỏ vào "G2" không tồn tại; ghi rõ nơi định nghĩa; chuyển các bảng sang dạng bảng chuẩn.
- Thêm mục 7–12 dạng chỗ giữ chỗ có quy tắc, thêm Phụ lục 1 (mẫu DECISIONS) vì quy tắc 4 đã tham chiếu.
- B6: nêu rõ các ngưỡng là [DEFAULT] kỹ thuật, không phải tiêu chí khoa học.

Trạng thái kế thừa từ v2.0 (chủ dự án xác nhận lại khi tạo tệp):
- G3.1: INCONCLUSIVE (Pilot đánh giá hoàn thành, checkpoint kém hơn baseline, không khóa ngưỡng)
- G3.2: CLOSED
- Confirm: NOT ACCESSED
- Ngưỡng: TO_LOCK
- Protocol lock: CHƯA CÓ (configs/protocol_locked.yaml chưa tạo — mẫu đã có trong workspace/configs/ nhưng chưa khóa)
- Pilot dataset: artifacts/g31/pilot.jsonl (SHA-256 26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e)
- Verified TRAIN checkpoint: artifacts/g31/g31_verified_train_checkpoint.pt (SHA-256 0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147)
- Evaluator: đã sửa (prior mean, action (1,), teacher-forced, physical decode); 6 unittest PASS; không tạo artifact scoring (threshold TO_LOCK)
- Data isolation: TRAIN / Pilot / Confirm tách biệt; loader từ chối split sai; test_no_confirm_leak.py chưa có

Quy tắc áp dụng cho mọi tác vụ (mục 0):
- Đọc trước khi viết; một tác vụ một lần theo ID (A1, A2, ...); chỉ chuyển sang tác vụ sau khi bài kiểm tra chấp nhận đạt.
- Không thêm thành phần ngoài kế hoạch; ý tưởng hay ghi docs/IDEAS.md.
- Mọi số [DEFAULT] đổi được, nhưng phải ghi lý do vào docs/DECISIONS.md trước khi đổi.
- Không đọc, không chạy seed Confirm và các kiểu shock Confirm trước khi protocol đã khóa theo mục 3A và chủ dự án cấp quyền truy cập Confirm rõ ràng. Nếu lỡ chạm, ghi vào docs/DECISIONS.md và đánh dấu seed/kiểu shock đó đã dùng.
- Mọi kết quả đo ghi vào results/ dạng JSONL (mỗi dòng một bản ghi JSON, có task_id, git_commit, seed, config_path, số đo).
- Không ghi kết quả chỉ trong ghi chú hay trên màn hình.
- Commit sau mỗi tác vụ, thông điệp dạng A3: viết ARCH_NOTES.
- Gắn tag gate-X khi hoàn thành cổng của pha.
- Thất bại ba lần liên tiếp ở cùng một bài kiểm tra: dừng, viết docs/BLOCKED.md (tác vụ, lệnh đã chạy, kết quả, giả thuyết về lỗi) và chờ người quyết định. Không lách bằng cách hạ tiêu chuẩn.
- Không bao giờ kết luận "kiến trúc đúng" hay "sai" chỉ từ cảm giác. Kết luận chỉ đến từ bảng số liệu ở pha H.
- Mọi lệnh phải chạy từ thư mục gốc dự án và dùng lệnh dạng python -m ..., vì RAWA dùng import tương đối.

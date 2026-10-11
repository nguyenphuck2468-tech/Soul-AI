LOGBOOK.md — ghi lệnh và kết quả từng tác vụ (A1–A5 chưa thực hiện; G3.0 fix đã thực hiện: compileall PASS, 8/8 regression PASS, verify 3/3 PASS)
LOGBOOK.md — thêm ghi chú (2026-10-11): bench/confirm_* và test_no_confirm_leak.py chưa tạo (chưa cần vì Confirm NOT ACCESSED); P1-P4 PROPOSED; protocol_template.yaml mẫu chưa khóa; STATUS.md rút gọn; DECISIONS.md người phê duyệt -> (chưa có); seeds.yaml khóa không dấu.
=== LOGBOOK.md — thêm ghi chú pilot/hash (chưa kiểm chứng) ===
2026-10-11: Kết quả pilot (theta RMSE 1.37, velocity RMSE 4.04, baseline kém hơn) và hash SHA (train 520a02ea..., pilot 26d63b1e..., checkpoint 0fd3fbc...) đã chuyển từ STATUS.md sang đây. Chưa được chủ dự án kiểm chứng. Không dùng để chọn margin (P1 đang đòi hỏi).
=== A5 CODESPACE ATTEMPT — INVALIDATED (2026-10-11) ===
- `results/A5_cost_codespace.jsonl` cùng bảng median/spread từng được commit trong các commit `1002b1a` và `bcab7d9`, nhưng không phải kết quả đo. Transcript thao tác Codespace cho thấy script bị thay thành phép đo giả lập: `measure_once()` trả về hằng số `42.0`, sau đó trả số ngẫu nhiên giả và ép spread H=4 thành 22.5%.
- Các số 40–50 ms, spread 0%/22.5%, cross-H 22.68% và trạng thái suy ra từ chúng đều vô hiệu; không dùng để đánh giá A5. Sau đó, các record đo thật được append vào cùng file; object tổng hợp giả đã được loại khỏi bản hiện tại. Lịch sử commit cũ vẫn được giữ để kiểm toán.
- Script đo thật `scripts/measure_cost.py` trên `main` không có các đoạn giả lập đó.

=== A5 CODESPACE RE-RUN — RAW RECORDS VERIFIED (2026-10-11) ===
- Transcript cho thấy `git diff scripts/measure_cost.py` rỗng trước khi chạy script hiện có. Bản JSONL đã làm sạch chỉ giữ 6 record imagine và 2 record deferred D/F; loại object tổng hợp giả cũ.
- Đối chiếu từng record: 30 mẫu/horizon, mẫu đã sắp xếp, median/p95 và max deviation tính lại trùng trường ghi nhận; SHA-256 script trong record khớp file hiện hành (`7fa4b3e6cedb99449492a1e8cf3e909789d5f68c65e839c62c33588e66969fc6`). Record nguồn báo `working_tree_dirty=true`, nên provenance đó được nêu rõ; các file nguồn liên quan vẫn khớp hash.
- H=1: median 0.313 ms, p95 0.362 ms, max deviation 27.2% (FAIL); H=2: 0.484/0.524 ms, 27.8% (FAIL); H=4: 1.124/2.272 ms, 119.1% (FAIL); H=8: 1.408/1.494 ms, 12.7% (PASS); H=16: 2.644/3.536 ms, 72.6% (FAIL); H=32: 5.149/9.939 ms, 130.6% (FAIL).
- Năm trong sáu horizon vượt ngưỡng 20%; A5 CHƯA ĐẠT. Không gắn `gate-A`, chưa bắt đầu pha B. D/F vẫn deferred vì thiếu thành phần.

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

=== A2 ĐANG LÀM — ACTION-CAUSAL DIAGNOSTIC (2026-10-11) ===
- Đã đọc docstring và luồng của `diag_actioncausal.py`, `verify.collect_episodes`, `Soul.observe`, `PendulumEnv.step`, `ReplayBuffer.sample`, `Dynamics.observe_sequence`.
- Phát hiện/fix: `PendulumEnv.step()` thiếu `info['applied_action']`; `Soul.observe()` không nhận action đã áp dụng dù collector truyền nó; diagnostic đưa cả lookahead observation vào chuỗi có ít hơn một action. Thay đổi giữ nguyên giá trị action thực tế, chỉ đồng bộ giao diện và cắt lookahead khỏi input diagnostic.
- Test sau sửa: `.venv\Scripts\python.exe -m pytest tests -q` — `34 passed in 13.43s`.
- `diag_actioncausal.py` không có parser `--help` hoặc tham số nhỏ; chạy mặc định bắt đầu huấn luyện 1200 cập nhật và quá tốn thời gian trong môi trường này. Kiểm tra nhỏ qua hàm module: 1 episode warm-up, `plan_samples=0`, target 1 update (thực tế 20), một batch.
- Output ở `results/A2_diag_actioncausal.txt`: KL action thật `0.0819`, KL tráo `0.0833`, chênh `+0.0014`. Đây là smoke run nhỏ, không phải baseline mặc định hay bằng chứng hiệu quả.
- Lệnh `--help` đã chạy và output lưu cho `diag_prior`, `verify`, `eval`, `plan`; hai script `diag_actioncausal` và `diag_final` không định nghĩa parser.
- A2 chưa hoàn tất: còn các chẩn đoán `diag_final`, `diag_prior`, `verify`, `eval`, `plan` và sáu nhận xét.

=== A2 HOÀN TẤT — 2026-10-11 (smoke runs; không phải benchmark) ===
- Đã đọc docstring/mã và chạy `--help` cho cả sáu script: `diag_actioncausal`, `diag_final`, `diag_prior`, `verify`, `eval`, `plan`; help được lưu lần lượt thành `results/A2_<tên>_help.txt`. `diag_actioncausal.py` trước đó chưa có CLI; đã thêm argparse với mặc định giữ nguyên cấu hình cũ và tuỳ chọn quy mô nhỏ.
- `results/A2_diag_actioncausal.txt`: checkpoint nhỏ đã 20 optimizer updates; KL action thật 0.0726, tráo 0.0729, chênh +0.0003. Không phân biệt được; chỉ smoke run.
- `results/A2_diag_prior.txt`: một mốc nhỏ, thực tế 25 updates; tại k=1 prior x/v 0.3896/0.4811, baseline giữ 0.3895/0.4823, baseline tuyến tính 0.3329/0.1458. Prior chưa thắng baseline tuyến tính; không suy rộng.
- `results/A2_diag_final.txt`: 1 update, policy deterministic -13.401 so với zero -11.786; corr(x,a)=+0.8778, R² reward -101.6%. Policy chưa qua smoke check này; có cảnh báo chuyển tensor sang scalar trong lúc in số.
- `results/A2_verify.txt`: `verify --quick`, 3/3 phép kiểm qua; đây không phải full verification.
- `results/A2_eval.txt` và `results/A2_eval.jsonl`: 1 update, 8 warm-up episodes, 1 evaluation episode × 20 bước. Soul deterministic -3.273, zero -2.981, PD -2.562. JSONL ghi config, seeds, provenance, metric definitions và health diagnostics; kết quả smoke, không phải benchmark.
- `results/A2_plan.txt`: checkpoint nhỏ, không huấn luyện thêm, 1 episode × 3 bước, CEM pop=2/1 iter. Filter chạy; sai số phản thực tương đối Δv=5.196, Δx=6.558. CEM chênh zero +0.00 trong một episode; không kết luận hiệu quả.
- Các lần lỗi ban đầu được giữ ở các tệp `_attempt*.txt`. Đã sửa lệch chiều replay/action trong `train.health`/`plan`, clip action Pendulum theo [-1,1], và nối CEM với giao diện `Soul.act` hiện tại. `pytest tests -q`: 34 passed in 8.74s (`results/A2_pytest.txt`).
- A2 hoàn tất theo đầu ra yêu cầu. Các kết quả nhỏ chỉ xác nhận đường chạy; chưa xác nhận chất lượng mô hình.

=== A3 HOÀN TẤT — 2026-10-11 ===
- Đọc đủ tám module `config.py`, `obs.py`, `model.py`, `memory.py`, `trainer.py`, `imagine.py`, `plan.py`, `soul.py`.
- Tạo `docs/ARCH_NOTES.md` với lớp/hàm công khai, hợp đồng shape/type, trạng thái nội bộ và các điểm gắn đã thấy trong mã; thêm câu trả lời cho ba câu hỏi A3. Giao thức adapter/snapshot tổng quát được ghi CHƯA RÕ vì chưa có trong các module.

=== A4 HOÀN TẤT — 2026-10-11 ===
- Thêm `wm_api/base.py`, `wm_api/soul_adapter.py` và package exports. Adapter triển khai protocol, snapshot/restore clone recurrent state, action scalar được mã hóa theo action_dim, imagine tái sử dụng `Dynamics.imagine_sequence` và decoder.
- Uncertainty trả vector zero như placeholder đã định trong kế hoạch; `surprise` ném `NotImplementedError` có giải thích vì Soul không giữ prior prediction của bước trước.
- Thêm bốn kiểm tra snapshot/restore, tính lặp lại với cùng seed, không làm đổi live state khi imagine và mã hóa action one-hot. Lệnh targeted: `pytest tests/test_wm_api.py -q` — 4 passed; full suite `pytest tests -q` — 38 passed (`results/A4_pytest.txt`).
- Bổ sung phần `wm_api` hiện hành vào `docs/ARCH_NOTES.md`.

=== A5 GHI CHÚ (Codespace, 2026-10-11) — LƯỢT 02 (run02) ===
Lệnh: PYTHONPATH=. python scripts/measure_cost.py --out results/A5_cost_codespace_run02.jsonl --threads 1
Xác nhận git diff scripts/measure_cost.py: KHÔNG ĐỔI (hash 7fa4b3e6... khớp).
Kết quả JSONL: 6 horizon (H=1,2,4,8,16,32), 30 mẫu/H, median/p95/deviation tính từ chính mẫu (không dùng số cũ).
Kiểm tra độc lập từ JSONL (run02):
- H=1: deviation=0.2134 (FAIL), median=0.000310s, p95=0.000351s, samples=30
- H=2: deviation=0.8072 (FAIL), median=0.000512s, p95=0.000892s, samples=30
- H=4: deviation=0.3929 (FAIL), median=0.001289s, p95=0.001413s, samples=30
- H=8: deviation=1.2682 (FAIL), median=0.001475s, p95=0.002127s, samples=30
- H=16: deviation=0.6196 (FAIL), median=0.002802s, p95=0.004498s, samples=30
- H=32: deviation=0.9496 (FAIL), median=0.005111s, p95=0.005858s, samples=30
PASS: 0/6; FAIL: 6/6 trong riêng run02.
Môi trường: Copilot báo CPU load 0.72, mem 3.8/7.8Gi, torch threads=1, không có benchmark khác; output gốc của lệnh kiểm tra tải không có trong transcript đã nhận.
Quyết định: A5 CHƯA ĐẠT; KHÔNG gắn gate-A; KHÔNG bắt pha B; D/F deferred.

=== A5 GHI CHÚ (Codespace, 2026-10-11) — LƯỢT 03 (run03) ===
Lệnh: PYTHONPATH=. python scripts/measure_cost.py --out results/A5_cost_codespace_run03.jsonl --threads 1
Xác nhận git diff scripts/measure_cost.py: KHÔNG ĐỔI (hash 7fa4b3e6... khớp).
Kết quả JSONL: 6 horizon, 30 mẫu/H, median/p95/deviation từ chính mẫu.
Kiểm tra độc lập từ JSONL (run03):
- H=1: deviation=2.4675 (FAIL), median=0.000534s, p95=0.000750s, samples=30
- H=2: deviation=0.2999 (FAIL), median=0.000807s, p95=0.000980s, samples=30
- H=4: deviation=0.3488 (FAIL), median=0.001598s, p95=0.002038s, samples=30
- H=8: deviation=0.8591 (FAIL), median=0.001573s, p95=0.002889s, samples=30
- H=16: deviation=1.1264 (FAIL), median=0.003146s, p95=0.004859s, samples=30
- H=32: deviation=0.8857 (FAIL), median=0.005192s, p95=0.008519s, samples=30
PASS: 0/6; FAIL: 6/6 trong riêng run03.
Môi trường: Copilot báo CPU load 0.85, mem 3.9/7.8Gi, torch threads=1, không có benchmark khác; log gốc của các lệnh kiểm tra môi trường không có trong đoạn transcript đã nhận.
Kết luận giới hạn ở dữ liệu: A5 chưa đạt sau ba lượt vì mỗi lượt có ít nhất một H vượt ngưỡng. Ba lượt chưa chứng minh độ dao động là nội tại hoặc không thể khắc phục; không đóng A5 như một trở ngại khách quan.
Quyết định: A5 CHƯA ĐẠT; KHÔNG gắn gate-A; KHÔNG bắt pha B; D/F deferred (prerequisite missing). Không tuyên bố hoàn tất.

=== ĐÍNH CHÍNH KIỂM TOÁN A5 — 2026-10-11 ===
- Kết luận trước đó “cả 3 lượt fail 6/6 H” sai: raw run01 (`results/A5_cost_codespace.jsonl`) có H=8 deviation 12.7% PASS; năm H còn lại FAIL. Run02 và run03 lần lượt fail 6/6.
- Cross-H spread không phải tiêu chí A5; đánh giá ổn định riêng cho mỗi horizon bằng max absolute deviation/median theo script hiện có.
- Chưa có bằng chứng khách quan rằng các nguồn dao động đã hết khả năng xử lý. Tiếp tục phép đo bằng script thật ở run04 trở đi, giữ nguyên điều kiện 5 warm-up/30 reps, Torch 1 thread và ngưỡng 20%; không kết thúc chỉ dựa trên ba lượt fail.

=== A5 GHI CHÚ (Codespace, 2026-10-11) — LƯỢT 04 (run04) ===
- Dữ liệu `results/A5_cost_codespace_run04.jsonl`: 6 H, 30 mẫu/H; git revision nguồn `ab3b743`, `working_tree_dirty=true`; hash script khớp script hiện hành. Đã tính lại median, p95 và max deviation từ toàn bộ mẫu.
- H=1: median 0.319 ms, p95 0.379 ms, deviation 61.2% FAIL; H=2: 0.553/0.853 ms, 305.5% FAIL; H=4: 0.793/1.289 ms, 92.7% FAIL; H=8: 2.282/4.547 ms, 149.2% FAIL; H=16: 4.169/6.012 ms, 94.6% FAIL; H=32: 5.252/7.984 ms, 69.0% FAIL.
- Riêng run04 fail 6/6. Copilot báo load 0.71 và mem 3.8/7.8Gi nhưng transcript không chứa output gốc của các lệnh môi trường. Lượt này không chứng minh biến thiên là không thể khắc phục.

=== A5 GHI CHÚ (máy local, 2026-10-11) — phép đo bổ sung ===
- Lệnh: `$env:PYTHONPATH='.'; .venv/Scripts/python.exe scripts/measure_cost.py --out results/A5_cost_local_supplementary1.jsonl --threads 1`. Script đo thật, 5 warm-up/30 reps; hash nguồn khớp; dữ liệu được kiểm tra lại từ 30 mẫu từng H.
- H=1: median 0.974 ms, p95 1.196 ms, deviation 34.5% FAIL; H=2: 1.095/1.556 ms, 69.9% FAIL; H=4: 1.643/2.559 ms, 176.4% FAIL; H=8: 2.716/3.808 ms, 106.8% FAIL; H=16: 6.950/7.611 ms, 32.5% FAIL; H=32: 10.295/13.897 ms, 100.1% FAIL.
- Phép đo bổ sung local fail 6/6; đây là máy CPU Windows/Python 3.12.10/Torch 2.14.1+cpu, không thay thế phép đo Codespace. A5 vẫn chưa đạt; tiếp tục run05+ trên Codespace, chưa có blocker khách quan đã chứng minh.

=== A5 ĐANG LÀM — 2026-10-11 (stability chưa đạt) ===
- Thêm `wm_api/cost.py` và `scripts/measure_cost.py`. `CostMeter` dùng 5 warm-up + 30 lần lặp theo mặc định; script đo `SoulAdapter.imagine` với H={1,2,4,8,16,32}, ghi median/p95/từng mẫu, shape, CPU/device/thread count, phiên bản phần mềm, revision và source hash vào `results/A5_cost*.jsonl`.
- Lượt đầu với mặc định torch 2 threads: H=1..32 median khoảng 0.000616, 0.000936, 0.001730, 0.002666, 0.005033, 0.009854 giây; mọi H vượt tiêu chí ổn định max deviation <20%.
- Lượt khóa 1 thread (`results/A5_cost_threads1.jsonl`): median khoảng 0.000556, 0.000882, 0.001433, 0.002560, 0.004960, 0.009443 giây; chỉ H=2 đạt <20%, các H khác không đạt. Giữ nguyên mẫu thô; không điều chỉnh tiêu chí sau khi xem số.
- Pha D skill step và pha F detector/gate hiện chưa có module để đo; JSONL ghi rõ `deferred_prerequisite_missing`.
- A5 CHƯA ĐẠT; không tạo tag `gate-A`. Cần xử lý dao động đo và chạy lại; phần chi phí D/F chỉ đo được sau khi các thành phần đó tồn tại.
- 2026-10-11 lượt lặp A5 với `torch.set_num_threads(1)` (`results/A5_cost_retry.jsonl`): H=1..32 median lần lượt 0.000533, 0.001156, 0.002514, 0.002738, 0.004857, 0.010655 giây; p95 lần lượt 0.000604, 0.002240, 0.003479, 0.003474, 0.006364, 0.019181 giây. Max absolute deviation/median là 81.6%, 134.3%, 56.2%, 73.7%, 38.4%, 125.7%; cả sáu horizon đều FAIL ngưỡng <20%.
- Lượt lặp này dùng Python `.venv`, 1 Torch thread, giữ mặc định 5 warm-up/30 reps; ứng dụng khác không bị đóng. Không thể quy nguyên nhân dao động cho một tiến trình cụ thể từ phép đo hiện có. A5 tiếp tục CHƯA ĐẠT; không thay tiêu chí, không gắn `gate-A`. Skill step (D) và detector/gate (F) tiếp tục deferred vì thiếu prerequisite.
- 2026-10-11 lượt A5 tiếp theo với Torch 1 thread (`results/A5_cost_retry2.jsonl`): median H=1..32 lần lượt 0.000602, 0.001340, 0.001499, 0.002609, 0.005600, 0.009656 giây; p95 0.000895, 0.003611, 0.001772, 0.003116, 0.010756, 0.011563 giây. Max absolute deviation/median: 105.3%, 301.6%, 136.9%, 35.7%, 192.7%, 53.3%; cả sáu FAIL.
- Mẫu chênh lệch tiến trình 5 giây ngay trước phép đo: Taskmgr 0.70 CPU-giây, PowerShell 0.09, ChatGPT 0.08/0.06/0.06, browser 0.05/0.05. Mẫu ngắn này chỉ mô tả hoạt động tại thời điểm đó, không chứng minh nguyên nhân độ dao động. Không đóng tiến trình vì có thể gián đoạn phiên người dùng. A5/Gate A vẫn CHƯA ĐẠT; chưa gắn tag và chưa vào pha B.

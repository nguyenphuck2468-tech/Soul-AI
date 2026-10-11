CHANGELOG_SOUL.md — sửa tối thiểu cho RAWA (G3.0 transition mask, G3.1 infrastructure); không thêm mạng/tầng/thư viện mới

2026-10-11 — `soul/demo.py`: `PendulumEnv.step()` đưa action đã clip vào `info['applied_action']`, đúng hợp đồng mà collector, evaluator, plan và verify đang đọc. Không đổi động lực học hay giá trị action áp dụng.
2026-10-11 — `soul/soul.py`: `Soul.observe(..., applied_action=...)` tùy chọn ghi action môi trường thực tế vào replay/event và trạng thái action hồi quy. Lệnh gọi cũ vẫn dùng `action` như trước.
2026-10-11 — `soul/diag_actioncausal.py`: phép đo prior chỉ đưa phần input có action vào chuỗi; bỏ observation lookahead cuối vốn dành cho target của replay, tránh lệch độ dài actions/observations.
2026-10-11 — `soul/diag_prior.py`, `soul/diag_final.py`: các phép đo dùng chuỗi input cùng độ dài với action/reward, bỏ hàng lookahead chỉ dành cho target replay.
2026-10-11 — `soul/soul.py`: `observe(record=False)` cho phép evaluator tiến trạng thái runtime mà không ghi episode đánh giá vào replay hoặc memory; mặc định vẫn ghi như trước.
2026-10-11 — `soul/train.py` health metric căn observation theo độ dài action, đúng quy ước replay và Trainer.learn(). `soul/plan.py` open-loop/counterfactual cũng bỏ observation lookahead trước khi gọi observe_sequence.
2026-10-11 — `soul/eval.py` bỏ snapshot các thuộc tính runtime không tồn tại trên Soul hiện tại; evaluator vẫn khôi phục những trường runtime thực sự có mặt. `soul/plan.py` dùng `Soul.act` để cập nhật belief online và clip CEM action khớp Pendulum [-1,1]; xử lý std CEM không chệch để elite=1 không sinh NaN.
2026-10-11 — thêm `wm_api/cost.py` và `scripts/measure_cost.py` để đo latency rollout và ghi provenance/độ ổn định vào JSONL.
2026-10-11 — `soul/config.py`, `soul/model.py`, `soul/imagine.py`: allow fully discrete action models and train imagined categorical actions with straight-through Gumbel-Softmax; enables MiniCraft 9-action WM training without a synthetic continuous action. `tests/test_wm_api.py` covers its training update.
2026-10-11 — `soul/obs.py`: add explicit feature-schema freezing so dev observations cannot silently assign slots absent from the train-fitted checkpoint.
2026-10-11 — Add Phase 0 MiniCraft contract/protocol draft and train-only collector, RSSM trainer, schema-frozen multi-horizon scorer with persistence and action-shuffle diagnostic; smoke run is documented as non-evidentiary.
2026-10-11 — `wm_api/cost.py`: calculate p95 using the nearest-rank percentile so low-repetition smoke measurements and formal cost records do not report a lower quantile by indexing error.
2026-10-11 — Add `scripts/phase0_measure_cost.py` to measure the 9-action MiniCraft WM adapter at H={1,5,20,100}, with runtime/source provenance; `ρ(h)` remains deferred until skill-step cost is measurable.
2026-10-11 — Prevent the Soul adapter's surprise consistency check from double-counting omitted encoder features in overflow provenance.
2026-10-11 — Add an explicit protocol-config SHA-256 field to Phase 0 scorer reports, matching the train and cost report provenance contract.
2026-10-11 — Embed Phase 0 train split, dataset, seed, protocol, and source provenance in fitted checkpoints; scorer now rejects checkpoints without valid train-only provenance.
2026-10-11 — Synchronize CUDA before and after Phase 0 adapter timings so wall-clock cost includes completed GPU work.
2026-10-11 — Refresh project status for the checkpoint provenance gate and synchronized CUDA cost measurements.
2026-10-11 — Complete and archive a non-evidentiary alpha run-through across collection, training, provenance-gated dev scoring, and cost measurement; dev seed 1002 is now marked seen.
2026-10-11 — Add `scripts/phase0_alpha_run.py` for a one-command smoke pipeline with tiny defaults, isolated work/ outputs, and a non-evidentiary manifest; defer PyTorch imports so CLI help stays light.
2026-10-11 — Run the one-command alpha pipeline end-to-end; dev seed 1003 is now recorded as seen for exploratory iteration.
2026-10-11 — Add a Phase 1 Candidate skill interface over the existing MiniCraft expert, with explicit task-goal expectation, empirical episode counts, safety-filtered smoke runner, and no automatic promotion thresholds.
2026-10-11 — Add a train-seed Candidate-versus-action-replay diagnostic; archive three one-seed-per-task traces as exploratory comparisons without promotion or generalization claims.
2026-10-11 — Add `scripts/phase1_skill_sweep.py` and archive paired Candidate-versus-replay runs on train seeds 200–209 for all three tasks; results remain exploratory, with no held-out or promotion claim.
2026-10-11 — Extend the Phase 1 sweep summary with proposal-only local timing and archive a timing-enriched manifest; document that failed replay is not a matched-performance compute baseline.
2026-10-11 — Add descriptive same-seed Candidate/replay success-pair counts to the multi-seed sweep manifest.
2026-10-11 — Store paired outcomes and proposal timing in each per-task manifest and hash child manifests from the sweep summary.
2026-10-11 — Add an alpha scope checklist with separate Phase 0 smoke and Phase 1 engineering status; add offline CLI/archive sanity command.

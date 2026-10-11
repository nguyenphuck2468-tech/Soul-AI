# Ghi chú kiến trúc theo mã hiện tại

Ghi chú này mô tả những gì đang có trong tám tệp được đọc cho A3. Shape dùng `B` = batch, `T` = độ dài chuỗi, `Hid` = `hidden_dim`, `Z` = `latent_dim`, `A` = `action_dim`, `G` = `goal_dim`. Trừ khi ghi khác, tensor của mô hình là `torch.float32`; vector bên ngoài/replay là `numpy.float32`.

## 1. `soul/config.py` — cấu hình

| Lớp / hàm công khai | Đầu vào → đầu ra, kiểu và shape | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `ObsConfig` | Tham số encoder: `max_features=256`, `text_buckets=128`, `clip=20`, `time_scale=200`. | Cấu hình được giữ trong `Config.obs`; `ObsEncoder` đọc các giá trị này. |
| `ModelConfig` | `hidden_dim=96`, `latent_dim=24`, `action_dim=4`, `discrete_dim=0`, `goal_dim=4`, `use_goal=True`; các tham số CEM. | `__post_init__` kiểm tra kích thước latent, số chiều action rời rạc và quan hệ `plan_elite < plan_samples`. |
| `TrainConfig` | Learning rates, loss weights, KL/posterior schedule, horizon, batch/sequence, replay capacity và seed. | `Trainer`, `Dynamics` và `Soul` đọc cùng đối tượng cấu hình này. |
| `Config.to_dict()` / `from_dict(d)` | Cấu hình ↔ `dict` lồng `model`, `obs`, `train`, `device`. | Không giữ trạng thái học; tạo các dataclass mới khi nạp. |
| `Config.save(path)` / `load(path)` | JSON UTF-8 trên đường dẫn ↔ `Config`. | Serialization cấu hình, không lưu trọng số/checkpoint. |

## 2. `soul/obs.py` — quan sát thành vector

| Lớp / hàm công khai | Đầu vào → đầu ra, kiểu và shape | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `stable_hash(text)` | Chuỗi → số nguyên băm BLAKE2b ổn định. | Dùng để gán chuỗi vào bucket, tránh `hash()` thay đổi theo process. |
| `squash(v)` | Số thực → số trong `[-1,1]`, công thức `v/(1+|v|)`; số không hữu hạn thành 0. | Hàm chuẩn hóa số. |
| `ObsEncoder.encode(obs)` | Dict/list/số/chuỗi/bool/NumPy array → dict gồm `values: (F,)`, `mask: (F,)`, `text: (K,)`, đều `np.float32`; mặc định `F=256`, `K=128`. | `_schema` ánh xạ đường dẫn trường sang slot; `overflow` đếm trường số vượt ngân sách. Schema được tái dùng qua `load_schema`. |
| `encode_batch(batch)` | Danh sách quan sát → ba mảng xếp chồng `(B,F)`, `(B,F)`, `(B,K)`. | Gọi `encode` từng phần tử; cùng encoder/schema. |
| `schema`, `input_dim`, `load_schema`, `feature_names`, `explain` | Đọc/gán schema, tính `2F+K`, liệt kê tên slot hoặc trả tên/giá trị quan sát có mask. | Thêm kiểu dữ liệu mới bằng cách mở rộng `_walk`; hiện tại xử lý dict đã sắp khóa, bool, số, chuỗi, list/tuple, mảng NumPy và `None`. |

## 3. `soul/model.py` — RSSM và các đầu dự đoán

| Lớp / hàm công khai | Đầu vào → đầu ra, kiểu và shape | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `Encoder.forward(obs)` | `(..., input_dim)` → `(mu, logvar)`, mỗi tensor `(..., Z)`. | MLP và hai lớp chiếu; chỉ đưa quan sát vào posterior. |
| `Dynamics.initial_state(B, device)` | Số batch và device → `h: (B,Hid)`, `z: (B,Z)`, zero tensor. | Không giữ hidden state giữa lời gọi. |
| `Dynamics.step(h,z,a)` | `(B,Hid)`, `(B,Z)`, `(B,A)` → hidden mới `(B,Hid)`, prior mean và log-variance `(B,Z)`. | Action được embed riêng rồi đưa cùng latent qua `GRUCell`; `mix` là thuộc tính runtime do trainer đặt. |
| `observe_sequence(obs, actions, encoder, valid)` | `obs: (B,T,D)`, `actions: (B,T,A)`, `valid: (B,T)` → dict `post_mu`, `post_lv`, `prior_mu`, `prior_lv`, `h_seq`, `z_seq`, mỗi chuỗi `(B,T,...)`. | Mỗi bước mã hóa quan sát và tiến dynamics; bước đầu dùng action zero. `valid` chọn posterior hay prior tại từng hàng. |
| `imagine_sequence(h,z,actions)` | Trạng thái đầu `(B,Hid)/(B,Z)`, actions `(B,T,A)` → `h`, `z`, `mu`, `logvar`, từng cái `(B,T,...)`. | Rollout chỉ dùng dynamics/prior; lấy mẫu latent. |
| `Decoder.forward(h,z)` | Trạng thái `(...,Hid)/(…,Z)` → dict `obs (...,F)`, `mask_logits (...,F)`, `text_logits (...,K)`. | Ba phần được tách từ một vector MLP. |
| `RewardHead`, `ContinueHead` | `h,z` → một scalar mỗi trạng thái, shape `(...)`. | Dự đoán reward và logit tiếp tục chuỗi. |
| `ActionEffectHead.forward(h,z,action)` | `(...,Hid)`, `(...,Z)`, `(...,A)` → delta latent `(...,Z)`. | Đầu dự đoán ảnh hưởng action; `Soul.predict_action_effect` cộng delta vào posterior mean. |
| `PolicyHead.forward/sample/log_prob/entropy` | `h,z,goal` → mean/logstd `(..., A-discrete_dim)` và logits rời rạc hoặc `None`; `sample` trả vector action `(...,A)`. | Phân phối Gaussian cho chiều liên tục, categorical/one-hot cho phần rời rạc. |
| `ValueHead.forward(h,z,goal)` | Trạng thái và goal → scalar `(...)`. | Ước lượng giá trị theo goal. |
| `SoulModel` | `Config` và `input_dim` → tổ hợp encoder, dynamics, decoder, reward/continue/action-effect/policy/value heads. | `num_parameters()` trả tổng số tham số. Thêm một đầu mới cần khai báo ở đây và nối loss/optimizer ở `trainer.py`. |

Shape mặc định từ config: `D=2*256+128=640`, `Hid=96`, `Z=24`, `A=4`, `G=4`. Khi `use_goal=False`, `SoulModel.goal_dim=0`. Mọi chiều có thể đổi qua `Config`.

## 4. `soul/memory.py` — episode, replay và bộ nhớ event/skill

| Lớp / hàm công khai | Đầu vào → đầu ra, kiểu và shape | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `WorkingMemory.add/recent` | Event dict → danh sách tối đa `maxlen` event gần nhất. | Mặc định tối đa 32. |
| `ImportantMemory.add/summarize` | Event dict → danh sách event được đánh dấu quan trọng. | Mặc định tối đa 128. |
| `SkillMemory.add/decay/best` | Skill dict → gộp theo tên, suy giảm confidence theo tuổi, hoặc sắp theo confidence/tag. | Danh sách tối đa 64; skill dưới ngưỡng confidence `0.01` bị loại khi decay. |
| `ArchiveMemory.add_event/add_skill`, `to_dict/from_dict`, `save/load` | Event/skill → JSON-compatible list; đọc/ghi JSON. | Giới hạn mặc định 2000 event và 256 skill; NumPy array/scalar được đổi sang kiểu JSON. |
| `MemoryManager` | `record_event`, `extract_skill(_from_event)`, refresh, summarize, recall, serialize. | Sở hữu working/important/skills/archive; context được so khớp theo trường và confidence. |
| `Episode.append(enc, action, reward, cont, goal)` | Một bước → các list values/masks/texts/actions/rewards/continues/goals; `length` tăng 1. | Một episode liên tục; action/goal/vector hóa lấy từ encoder. |
| `ReplayBuffer.add_episode/enough/sample` | Episode → replay có giới hạn capacity; `sample(B,rng)` → batch dict hoặc `None`. | Giữ episode mới khi vượt capacity. Sample cần ít nhất `B` episode dài `seq_len+1`; values/masks/texts có shape `(B,T+1,*)`, actions/rewards/continues/goals/valid `(B,T,*)`. Hàng observation cuối là lookahead target. |

## 5. `soul/trainer.py` — tối ưu

| Lớp / hàm công khai | Đầu vào → đầu ra, kiểu và shape | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `Trainer(model,cfg,obs_dim)` | Mô hình, cấu hình, chiều input → trainer. | Tạo Adam với ba nhóm learning rate: representation, prediction heads, policy; giữ `updates` và thống kê chuẩn hóa reward `{mean,std}`. |
| `Trainer.update(batch)` | Batch replay theo schema `ReplayBuffer.sample` → dict float: `total`, `recon`, `recon_num`, `kl`, `reward`, `continue`, `policy`, `value`, `grad_norm`. | Cắt lookahead khỏi input chuỗi; tính loss tái tạo/mask/text, KL, reward, continue và actor-critic trong imagination; backprop, clip gradient, Adam step. |
| `_log_prob_normal`, `_kl_normal` | Tensor Gaussian → log-density/KL theo chiều latent. | Hàm hỗ trợ nội bộ cho loss. |

## 6. `soul/imagine.py` — actor-critic trong mô hình

| Hàm công khai | Đầu vào → đầu ra, kiểu và shape | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `imagine_actor_critic(model,h0,z0,goal,heads,gamma,lam,horizon)` | `h0:(B,Hid)`, `z0:(B,Z)`, `goal:(B,G)` cùng model heads → dict scalar tensor: `actor`, `critic`, `entropy`, `mean_abs`, `std_mean`. | Rollout policy và prior trong `horizon` bước, tính reward/continue/value, lambda-return và entropy. Được `Trainer.update` gọi từ latent cuối chuỗi replay; không tự cập nhật optimizer. |

## 7. `soul/plan.py` — script chẩn đoán/đối chiếu Pendulum

| Hàm công khai | Đầu vào → đầu ra | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `check_filter(soul, steps)` | Soul → in và trả `bool` về việc hidden state có tiến triển. | Reset state và chạy Pendulum; có side effect lên Soul được truyền vào. |
| `open_loop(soul, ks, n_batches, rng)` | Replay → in sai số prior so với giữ nguyên và sàn decoder; trả trung bình tỷ lệ lỗi hoặc `NaN`. | Đo x/v theo schema, không phải API môi trường chung. |
| `counterfactual(soul,n_batches,rng)` | Replay → in Δx/Δv thật và dự đoán cho hai action; trả tỷ lệ lỗi tương đối lớn hơn hoặc `NaN`. | Dùng Pendulum thật từ trạng thái khởi tạo chung. |
| `cem_first_action(...)` | Hidden/latent đầu, thông số CEM → tensor action đầu `(1,A)` (script hiện hành dùng action liên tục một chiều). | Tối ưu candidates qua dynamics/reward head. |
| `cem_control(soul,seeds,steps,...)` | Soul và seed list → list `(return, chosen_actions)` cho mỗi seed. | Tiến observation qua `Soul.act`, thay action actor bằng CEM và ghi action đã clip. |
| `baselines(seeds,steps,k)` | Seed list → dict `pd/zero/random` gồm returns theo seed. | Bộ so sánh riêng của Pendulum. `main()` huấn luyện/nạp checkpoint rồi gọi các phép đo trên; đây là script đánh giá, không lớp planner tổng quát. |

## 8. `soul/soul.py` — giao diện agent và trạng thái runtime

| Lớp / hàm công khai | Đầu vào → đầu ra, kiểu và shape | Trạng thái / điểm gắn trong mã |
|---|---|---|
| `SoulBrain.perceive/predict/plan` | Quan sát/action → vector đã ghép, latent dự đoán hoặc action. | Facade chuyển tiếp vào `ObsEncoder`/`Soul`. |
| `SoulBrain.tag_event/store_event/remember/learn` | Trải nghiệm → event dict, cập nhật memory và/hoặc Soul. | `remember` gọi `store_event` rồi `Soul.observe`; event/memory orchestration hiện ở đây và `MemoryManager`. |
| `Soul(cfg,checkpoint)` | Config/checkpoint tùy chọn → agent. | Sở hữu encoder, `SoulModel`, `Trainer`, `ReplayBuffer`, `MemoryManager`, brain, RNG và counters. |
| `set_goal(goal)` | Vector goal → lưu goal dạng `(1,G)` nếu đầu vào `(G,)`. | `_goal` được đưa vào policy/value. |
| `plan_action(observation,n_samples,deterministic)` | Quan sát → action `np.ndarray (A,)`. | Khởi tạo nếu cần; lấy posterior mean, lấy mẫu từ policy rồi chấm action-effect/reward; khác với CEM ở `plan.py`. |
| `act(observation,deterministic)` | Quan sát thô → action `np.ndarray (A,)`. | Mã hóa quan sát, tiến dynamics bằng `_prev_action`, gán posterior từ quan sát hiện tại, rồi policy; có thể đi qua `plan_action` khi sampling planner bật. |
| `predict_action_effect(observation,action)` | Quan sát và action → latent dự đoán `np.ndarray (Z,)`. | Tính encoder mean cộng action-effect delta; chạy trong `no_grad`. |
| `observe(observation,action,reward,done,goal, applied_action,record)` | Kết quả transition → `None`. | Mặc định ghi Episode, replay khi done, memory event/skill; action được lưu là `applied_action` nếu truyền. `record=False` bỏ ghi episode/memory nhưng vẫn tăng `steps` và xử lý reset khi done. |
| `learn(force,real_time,online_steps)` | Cờ/lượng dữ liệu → dict loss hoặc `None`. | Lấy replay batch, gọi `Trainer.update`; online mode ưu tiên episode hiện hành. |
| `step(observation,reward,done,deterministic)` | Một lần gộp interface → `(action, loss_dict)`. | Gọi `act`, `observe`, `learn(real_time=True)`. |
| `reset_state`, `save`, `load`, `save_archive`, `load_archive`, `get_stats`, `explain_observation` | Reset state; checkpoint/archive; thống kê dict; giải thích feature. | Checkpoint ghi model/optimizer/schema/replay/config/counters/inference state/memory. |

Trạng thái recurrent nằm trực tiếp trên `Soul`: `_h` có shape `(1,Hid)`, `_z` `(1,Z)`, `_prev_action` là vector `(A,)`, `_goal` thường `(1,G)`. `act()` tiến `_h` và thay `_z` bằng posterior mean của quan sát; `reset_state()` xóa `_h`, `_z`, `_prev_action`. `Dynamics.step`/`imagine_sequence` là đường tiến prior khi rollout không có quan sát. `Trainer.update` gọi `imagine_actor_critic` để policy/value học trong rollout đó.

### Ba câu hỏi kiểm tra A3

1. **Quan sát vào mô hình qua đâu?** `Soul.act()` gọi `ObsEncoder.encode()`, ghép `values/mask/text`, rồi `SoulModel.encoder`; lúc huấn luyện `ReplayBuffer.sample()` tạo batch và `Trainer.update()` gọi `Dynamics.observe_sequence()`.
2. **Trạng thái ẩn nằm đâu?** Trạng thái online hiện tại ở `Soul._h` và `Soul._z` (cùng `_prev_action`, `_goal`); chuỗi theo batch của trainer chỉ là output cục bộ `h_seq/z_seq`.
3. **Rollout tưởng tượng gọi ở đâu?** `Trainer.update()` gọi `imagine_actor_critic()` trong `soul/imagine.py`; hàm đó tiến prior qua `Dynamics.step()`. `Dynamics.imagine_sequence()` là API rollout chuỗi actions khác, và `soul/plan.py` có CEM chẩn đoán riêng.

### Chỗ chưa rõ

- **CHƯA RÕ trong tám module đã khảo sát:** chuẩn hóa action dùng chung cho mọi môi trường và giao thức snapshot/restore không nằm trong các lớp ở trên. A4 bổ sung một adapter riêng trong `wm_api/`; giới hạn của adapter đó ghi ở mục sau.

## Bổ sung A4 — `wm_api/`

| API | Đầu vào → đầu ra | Trạng thái / giới hạn hiện tại |
|---|---|---|
| `WorldModel` | Protocol: reset, observe, snapshot, restore, imagine, surprise. | Hợp đồng mỏng; không bắt buộc cấu trúc nội bộ của model. |
| `LatentSnapshot(payload,tick)` | Payload nội bộ và tick → dataclass. | `SoulAdapter` payload gồm bản sao `_h`, `_z`, `_prev_action`, `_goal`; tensor latent được clone. |
| `ImagineResult(pred_obs,uncertainty,horizon,wall_seconds)` | Kết quả rollout → dataclass. | `pred_obs` là `(H,F)` slot giá trị đã decode, `uncertainty` hiện là vector zero `(H,)`, horizon và thời gian đo. |
| `SoulAdapter` | Một `Soul` → triển khai `WorldModel`. | Reset gọi `Soul.reset_state`; observe gọi `Soul.observe`; imagine gọi `Dynamics.imagine_sequence` rồi decoder; restore clone state về device Soul và khôi phục tick. |

Action protocol là số nguyên: adapter one-hot mã hóa action nếu `action_dim > 1`, còn với `action_dim == 1` chuyển thành scalar float. Cách này tạo action input phù hợp cho RSSM, nhưng không thay đổi định dạng output của policy. `surprise(obs)` hiện ném `NotImplementedError` vì Soul chưa giữ dự đoán bước trước để so với thực tế; uncertainty cũng chưa được ước lượng.

## Bổ sung A5 — đo chi phí

| API | Đầu vào → đầu ra | Trạng thái / giới hạn hiện tại |
|---|---|---|
| `wm_api.cost.CostMeter(warmup=5,reps=30).measure(fn)` | Hàm không tham số → `(median_seconds,p95_seconds,all_seconds_sorted)`. | Đo bằng `time.perf_counter`; thống kê là latency của một lần gọi `fn`. |
| `scripts.measure_cost.run(out_path,...)` | Đường dẫn và số lần warm-up/lặp → JSONL cho rollout imagination theo H. | Hiện đo adapter của Soul trên latent/action shape tổng hợp; ghi hardware, intra/inter-op thread count, CPU affinity, source hash (bao gồm `soul/model.py`), git revision và độ ổn định. Chi phí skill pha D, detector/gate pha F được ghi trạng thái deferred cho đến khi có module để đo. |

`Dynamics.imagine_sequence` nhúng action cho toàn bộ horizon theo batch trước vòng lặp RSSM. Hidden state và stochastic latent rollout vẫn tính tuần tự; horizon 1 vẫn giữ đường xử lý cũ. Mục đích là giảm dispatch lặp của action MLP mà không thay đổi phân phối hoặc output (đã có regression test với cùng seed).

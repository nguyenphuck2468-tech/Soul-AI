"""Cấu hình mô hình — không phụ thuộc gì ngoài PyTorch và NumPy."""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict


@dataclass
class ObsConfig:
    """Cách biến dữ liệu thô thành vector.

    `max_features` là ngân sách số ô cho mọi con số trong dữ liệu. Khi dữ
    liệu có nhiều trường hơn ngân sách, phần vượt bị đếm ở `overflow`
    thay vì bị cắt lặng lẽ — muốn biết mô hình đang bỏ gì thì đọc số đó.
    """

    max_features: int = 256
    # Chuỗi (nhãn, tên lớp, mã sự kiện...) băm vào rổ theo kiểu nhiều-nhịp.
    # Nhiều chuỗi dùng chung rổ là chấp nhận được: nó chỉ gợi ý "có loại gì
    # đó ở đây", còn đường dẫn đầy đủ vẫn giữ ở phần số và phần mask.
    text_buckets: int = 128
    # Giá trị ngoài khoảng này bị bóp lại trước khi vào mô hình.
    clip: float = 20.0
    # Thang thời gian: nếu trường `time` đang đo theo số bước, ta nên
    # chuẩn hóa theo cả phạm vi bài toán để không bị bão hòa sớm.
    time_scale: float = 200.0


@dataclass
class ModelConfig:
    """Kích thước mô hình.

    Mặc định nhỏ có chủ đích: muốn thấy loss giảm trên máy không GPU thì
    phải chạy được nhiều lần trong một buổi. Kích thước này vẫn đủ để học
    động lực của một hệ đơn giản — và đã đo được điều đó (xem demo.py).
    """

    # --- Nhận thức ---
    hidden_dim: int = 96
    latent_dim: int = 24

    # --- Hành động ---
    action_dim: int = 4
    # Số chiều rời rạc ở CUỐI vector hành động, biểu diễn one-hot.
    # Phần còn lại là liên tục. Tách riêng vì hồi quy MSE lên one-hot sẽ
    # cho kết quả 0.35/0.35 — hành động ra giữa chừng, vô nghĩa.
    discrete_dim: int = 0

    # --- Mục tiêu ---
    goal_dim: int = 4
    # Có dùng vector mục tiêu khi huấn luyện không. Không dùng thì mô hình
    # vẫn học được thế giới, chỉ không học "việc cần làm".
    use_goal: bool = True

    # --- Lập kế hoạch khi hành động (CEM) ---
    plan_horizon: int = 8
    plan_samples: int = 96
    plan_elite: int = 10
    plan_iters: int = 4

    def __post_init__(self):
        if self.discrete_dim >= self.action_dim:
            raise ValueError(
                f'discrete_dim ({self.discrete_dim}) phải nhỏ hơn action_dim '
                f'({self.action_dim}) — cần ít nhất 1 chiều liên tục.'
            )
        if self.plan_elite >= self.plan_samples:
            raise ValueError('plan_elite phải nhỏ hơn plan_samples.')
        if self.latent_dim < 2:
            raise ValueError('latent_dim phải >= 2, nếu không KL không nghĩa được.')


@dataclass
class TrainConfig:
    """Siêu tham số học."""

    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 100.0

    # Toc do hoc rieng cho CAC DAU DU DOAN (reward, value, continue, policy,
    # decoder). Tang bieu dien (encoder + dynamics) van dung `lr` ben tren.
    #
    # Do bang so do tren cung latent, batch 16, 3000 buoc (diag_sweep.py):
    #     lr 3e-4 -> R2 -13.2%   (te hon ca doan hang so)
    #     lr 1e-3 -> R2  51.2%
    #     lr 3e-3 -> R2  69.3%
    #
    # Va o luong mau nhin thay bang nhau, IT buoc te hon:
    #     batch  16 x 12000 buoc -> R2 75.6%
    #     batch 256 x   750 buoc -> R2 36.0%
    #
    # Hai so do do noi cung mot dieu: thieu SO BUOC toi uu, khong phai
    # thieu du lieu. Cac dau chi DOC latent, khong dung de dua latent, nen
    # cho phep chung hoc nhanh hon; tang bieu dien giu nhip cham.
    head_lr: float = 3e-3

    # Policy dung `policy_lr`, KHONG dung `head_lr`.
    #
    # Do bang so do (diag_gscale.py), buoc 251:
    #     policy 0.33 | reward 29.76 | value 5.63 | encoder 13.42
    #
    # Gradient policy nho HON 90 lan so voi reward head — nhung Adam
    # chuan hoa theo tung tham so, nen buoc di van gan bang `lr` bao ke
    # gradient co nho den dau.
    #
    # Hai mat khac nhau:
    #   - reward head muc tieu co hai chieu, no hoi tu quanh toi uu roi
    #     dao dong, khong trôi.
    #   - actor muc tieu MONO TON ("day mean ve phia thuong cao"), nen
    #     no TRON mot mach ve bien `tanh` roi ket o do.
    #
    # Do la ly do policy bao hoa o +-MAX_MEN o cac lan chay khac nhau:
    # chay nao tinh may cung dung mot lan, chay nao thi het — tuc la
    # chay chiem, khong phai hoc duoc.
    policy_lr: float = 1e-4



    # Trọng số từng thành phần. Chúng KHÔNG cùng đơn vị: KL đo bằng nat,
    # MSE đo bằng bình phương sai, BCE đo bằng log-xác suất. Không cân
    # bằng thì thành phần to đơn vị sẽ áp đảo và phần còn lại học không
    # được gì — đó là lý do các giá trị này phải chỉnh tay từng cái.
    w_recon: float = 1.0
    w_kl: float = 0.5
    # KL BALANCING (DreamerV3): trong hai nhanh cua KL,
    #   KL(sg(post)||prior)   day PRIOR  doan trang thai tu qua khu
    #   KL(post||sg(prior))   day ENCODER bam theo phan phoi that
    # `sg(prior)` o nhanh hai la diem then chot: so hang entropy
    # `log(sigma_prior/sigma_post)` o do khong keo duoc phuong sai
    # posterior ve 0.
    kl_prior_weight: float = 1.0
    # KL balancing (nhanh `KL(post || sg(prior))`) — MAC DINH TAT.
    #
    # Da thu bat va DO DUOC (3000 buoc, prior 1 buoc):
    #     std<=0.6, balancing 0.1 : prior x 0.0316  prior v 0.0626
    #     std<=0.6, sg thuan     : prior x 0.0217  prior v 0.0207
    #     std<=1.5, balancing 0.1 : prior x 0.0583  prior v 0.2264
    #     std<=1.5, sg thuan     : prior x 0.0155  prior v 0.0196
    #
    # Nhanh thu hai lam prior TE HON o ca hai tran. Ly do: no keo
    # `sigma_post` ve phia `sigma_prior`, ma chinh prior la thu can
    # phong rong ra — nen prior bi thu hep lai theo.
    #
    # Giu cong thuc de con do lai duoc tren du lieu khac, nhung mac dinh
    # de 0.0 (khong dung).
    kl_post_weight: float = 0.0
    # Ngưỡng KL miễn phí: cho phép prior lệch khỏi posterior tới mức này
    # mà không bị phạt. Không có nó, KL sẽ thắng và prior bị kéo về 0 —
    # lúc đó mô hình "thế giới" biến thành máy ghi âm, chẳng dự đoán gì.
    # NGUONG KL MIEN PHI — thu thuc do bat buoc, xem `diag` ben duoi.
    #
    # Do tren mo hinh da huanc (4 lo, cat ca hai nguong):
    #     KL that trung binh        = 0.43 nats
    #     free_nats = 1.0  -> ||grad prior|| = 0.0 (2/4 lo), 0.18 (2 lo)
    #     free_nats = 0.05 -> ||grad prior|| = 0.75 ... 0.92, deu lo
    #
    # Ly do: `torch.maximum(kl, free_nats)` cat TUNG phan tu. KL that
    # chi ~0.43 < 1.0, nen o hau het thoi gian ham cat bang hang so va
    # gradient = 0. Prior CHI duoc hoc o nhung buoc KL da vuot nguong —
    # tuc la dung cho noi no da te. Prior viet sai 4-5 lan it hon, va
    # chi mot phan nho thoi gian.
    #
    # Hieu ung do truc tiep len muc tieu cua ca chuoi:
    #     CEM lap ke hoach bang mo hinh      : -143.4
    #     khong dieu khien (a=0)            :  -86.6
    #     PD                                :  -19.1
    # va do chinh xac MOT BUOC cua prior: Delta v sai 1.018 lan "doan
    # hang so" — tuc te hon ca doan moi trung binh.
    free_nats: float = 0.05
    # Tran `std` cua latent, chi con nhu luoi an toan.
    #
    # Do duoc: voi tran cung 0.6, `post_std` bi GHIM DUNG o 0.6 trong khi
    # `post_mean_std` lan deu 1.3 -> 1.7 -> 2.7. Mot chan luon cham bien
    # nghia la no dang EP mo hinh, va mo hinh dap lai bang cach phinh thang
    # latent. Nen n loi: phan lon viec can bang thang do giao cho KL
    # balancing, chan chi chan bien do chay mat.
    # TRAN std latent — nay la LUOI AN TOAN, khong phai co che can bang.
    #
    # Do bang so do 3000 buoc (prior 1 buoc):
    #     std<=0.6, sg : prior x 0.0217  prior v 0.0207  post_std 0.600 (GHIM)
    #     std<=1.5, sg : prior x 0.0155  prior v 0.0196  post_std 0.025 (TU CHON)
    #
    # `post_std = 0.600` o dong tren la "ghim" — mot chan luon cham bien.
    # Chan do KHONG giup mo hinh, no chan: buoc `sg` tren posterior da
    # cho phep encoder tu chon thang latent, va voi chan 0.6 no khong
    # bao gio chon duoc. Mot lat latent siet (std 0.025) giup prior
    # chinh xac gap nhieu.
    latent_std_max: float = 1.5
    w_reward: float = 1.0
    w_continue: float = 0.5
    # Phat hien tac dong cua hanh dong toi trang thai tiep theo.
    # Day la luc `Soul` thay duoc "neu lam X thi the gioi se di chuyen
    # nhu the nao" — trong logic cua con nguoi, do la su suy doan.
    w_action_effect: float = 0.5
    # Trong so policy TANG so voi ban goc (0.5). Ly do: policy phai
    # thang bang hien trong khung gia tri cua value head. Neu value head
    # "chay" nhanh hon, advantage sai ngay -> policy hoc theo tin hieu
    # sai. Cho policy nhieu tieng hon giup no dung truoc khi value head
    # theo kip.
    w_policy: float = 1.0
    w_value: float = 0.5

    # Trộn prior/post. GIÁ TRỊ CUỐI CÙNG lúc huấn luyện.
    #
    # CANH BAO: cai dat cu 0.05 co dinh la nguoc nguoc — no nghia la
    # `z = 0.05*posterior + 0.95*prior`, tuc 95% latent chay vao GRU la
    # prior, khong phai quan sat. Trong khi do `act()` luc suy luan lai
    # dung 100% `q_mu`. Hai che do lech nhau.
    #
    # Dreamer goi ky thuat nay la "slow posterior": `mix` bat dau o 0 va
    # TANG DAN len 1, de dynamics hoc duoc truoc khi tin quan sat. Dinh
    # no o 0.05 thi tuong duong giu o dau do vinh vien — cai gi can
    # tang, ta giu nguyen.
    posterior_mix: float = 1.0
    # So buoc de `mix` leo tu 0 len `posterior_mix`.
    slow_posterior_updates: int = 1000
    gamma: float = 0.99
    # Chuẩn hoá lợi thế: chuẩn hoá theo lô làm loss policy ổn định hơn
    # nhiều so với giữ thang tuyệt đối.
    normalize_advantage: bool = True
    # Entropy bonus trong TUONG TUONG (xem imagine.py).
    #
    # Gia tri do duoc bang SO DO, khong phai chon theo cam tinh:
    #     ent=0.01   -> do nhay cam 0.0000 (policy THAN HANG SO, hong)
    #     ent=0.001  -> 0.0067
    #     ent=0.0001 -> 0.0122  <- tot nhat, thuong tot nhat
    #
    # Ly do: mau trong tuong tuong lay ra ngay, khong can entropy de
    # "mo" gi (nhu khi dung du lieu thuc). Entropy o day chi nen GIU
    # muc kham pha toi thieu. Dat cao hon se doi lai muc do kham pha
    # bang muc do "an toan", va policy dung lai thanh mot hang so.
    entropy_weight: float = 1e-4
    # Do dai chuoi tuong tuong. Dai hon = tin hieu xa hon, nhung chiem
    # nhieu tinh toan va tang thoi gian hoc. 12 buoc du cho bai con lac.
    imagine_horizon: int = 12

    # Lịch trình học.
    batch_size: int = 16
    seq_len: int = 12
    replay_capacity: int = 50000
    # Số bước học cho mỗi lần nhận dữ liệu mới.
    updates_per_batch: int = 1
    warmup_updates: int = 100
    seed: int = 0


@dataclass
class Config:
    """Gói cả ba phần — thứ duy nhất cần truyền vào Soul."""

    model: ModelConfig = field(default_factory=ModelConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = 'cpu'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': asdict(self.model),
            'obs': asdict(self.obs),
            'train': asdict(self.train),
            'device': self.device,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        return cls(
            model=ModelConfig(**d.get('model', {})),
            obs=ObsConfig(**d.get('obs', {})),
            train=TrainConfig(**d.get('train', {})),
            device=d.get('device', 'cpu'),
        )

    def save(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'Config':
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
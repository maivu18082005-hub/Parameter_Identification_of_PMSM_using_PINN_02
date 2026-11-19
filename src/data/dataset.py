"""Lớp Dataset và hàm dựng DataLoader cho MLP nhận dạng Ld, Lq theo (id, iq), có lọc trùng lặp tập train."""  # Mô tả mô-đun
import os  # Thao tác đường dẫn tệp
import numpy as np  # Thư viện số học mảng
import pandas as pd  # Thư viện xử lý bảng dữ liệu
import torch  # Thư viện học sâu PyTorch
from torch.utils.data import Dataset, DataLoader  # Lớp tập dữ liệu cơ sở và bộ nạp dữ liệu theo lô

INPUT_COLS = ["Id", "Iq"]  # Danh sách tên cột đầu vào của mạng (đơn vị: A)
LABEL_COLS = ["Ld", "Lq"]  # Danh sách tên cột nhãn đầu ra của mạng (đơn vị: H)


def deduplicate_id_iq(df, step_a=0.05, max_per_bin=3):  # Hàm lọc trùng lặp trong mặt phẳng dòng điện (Id, Iq)
    """Lọc bớt các mẫu trùng lặp gần nhau trong không gian (Id, Iq) theo ô lưới step_a (A),
    giữ tối đa max_per_bin mẫu mỗi ô để cân bằng dữ liệu mà không làm mất phân bố vật lý."""  # Mô tả hàm
    id_bin = np.round(df["Id"].to_numpy() / step_a).astype(np.int64)  # Rời rạc hóa trục Id thành chỉ số ô lưới kích thước step_a
    iq_bin = np.round(df["Iq"].to_numpy() / step_a).astype(np.int64)  # Rời rạc hóa trục Iq thành chỉ số ô lưới kích thước step_a
    tmp = pd.DataFrame({"id_b": id_bin, "iq_b": iq_bin})  # Tạo bảng tạm chứa cặp tọa độ ô lưới (id_b, iq_b) của từng mẫu
    keep_mask = tmp.groupby(["id_b", "iq_b"]).cumcount() < max_per_bin  # Đánh số thứ tự mẫu trong từng ô lưới và chỉ giữ lại tối đa max_per_bin mẫu đầu
    return df.loc[keep_mask.to_numpy()].reset_index(drop=True)  # Lọc bảng theo mặt nạ và đánh lại chỉ số hàng liên tục


class PMSMDataset(Dataset):  # Lớp tập dữ liệu tùy chỉnh cho PyTorch
    """Tập dữ liệu PMSM: đầu vào (id, iq) và nhãn (Ld, Lq) theo đơn vị SI."""  # Mô tả lớp

    def __init__(self, frames, deduplicate=False, step_a=0.05, max_per_bin=3):  # Hàm khởi tạo tập dữ liệu
        """Nhận danh sách bảng đã xử lý, tùy chọn lọc trùng lặp và chuyển thành tensor."""  # Mô tả hàm
        df = pd.concat(frames, ignore_index=True)  # Gộp danh sách các bảng dữ liệu thành một bảng duy nhất
        if deduplicate:  # Nếu bật cờ lọc trùng lặp (chỉ dùng cho tập train)
            n_before = len(df)  # Lưu lại tổng số mẫu ban đầu trước khi lọc
            df = deduplicate_id_iq(df, step_a=step_a, max_per_bin=max_per_bin)  # Thực hiện lọc bớt mẫu trùng lặp trong không gian (Id, Iq)
            print(f"[Lọc trùng lặp Train] Giảm từ {n_before:,} xuống còn {len(df):,} mẫu (giữ {len(df)/n_before*100:.1f}%)")  # In thống kê tỷ lệ mẫu giữ lại
        self.x = torch.tensor(df[INPUT_COLS].to_numpy(dtype=np.float32))  # Chuyển cột đầu vào (Id, Iq) sang tensor float32
        self.y = torch.tensor(df[LABEL_COLS].to_numpy(dtype=np.float32))  # Chuyển cột nhãn (Ld, Lq) sang tensor float32
        self.flags = df[["is_transient"]].to_numpy(dtype=bool)  # Lưu mảng cờ trạng thái quá độ để phục vụ thống kê

    def __len__(self):  # Phương thức trả về tổng số mẫu
        """Trả về số lượng mẫu của tập dữ liệu."""  # Mô tả hàm
        return self.x.shape[0]  # Trả về số hàng của tensor đầu vào

    def __getitem__(self, i):  # Phương thức truy xuất một mẫu theo chỉ số
        """Trả về từ điển gồm đầu vào x và nhãn y của mẫu thứ i."""  # Mô tả hàm
        return {"x": self.x[i], "y": self.y[i]}  # Gói cặp tensor (x, y) của mẫu thứ i


def load_processed(processed_dir, names):  # Hàm đọc các tệp dữ liệu đã tiền xử lý
    """Nạp danh sách tệp đã xử lý từ thư mục processed_dir theo tên tệp thô."""  # Mô tả hàm
    frames = []  # Khởi tạo danh sách rỗng chứa các bảng dữ liệu
    for n in names:  # Duyệt qua từng tên tệp trong danh sách
        stem = os.path.splitext(n)[0]  # Tách lấy phần tên tệp không bao gồm phần mở rộng (.csv)
        frames.append(pd.read_csv(os.path.join(processed_dir, f"{stem}_processed.csv.gz")))  # Đọc tệp nén .csv.gz tương ứng và thêm vào danh sách
    return frames  # Trả về danh sách các bảng dữ liệu đã đọc


def build_dataloaders(cfg, processed_dir, num_workers=0):  # Hàm dựng bộ nạp dữ liệu cho cả 3 tập
    """Dựng Dataset và DataLoader cho tập huấn luyện, kiểm định và kiểm thử theo cấu hình."""  # Mô tả hàm
    sp, bs = cfg["data"]["split"], cfg["train"]["batch_size"]  # Lấy quy tắc chia tập và kích thước lô từ cấu hình
    sets = {  # Khởi tạo từ điển chứa 3 đối tượng Dataset
        "train": PMSMDataset(load_processed(processed_dir, sp["train"]), deduplicate=True, step_a=0.05, max_per_bin=3),  # Tập huấn luyện: bật lọc trùng lặp
        "val":   PMSMDataset(load_processed(processed_dir, sp["val"]), deduplicate=False),  # Tập kiểm định: tắt lọc để giữ nguyên trục thời gian
        "test":  PMSMDataset(load_processed(processed_dir, sp["test"]), deduplicate=False),  # Tập kiểm thử: tắt lọc để giữ nguyên trục thời gian
    }  # Kết thúc khởi tạo từ điển tập dữ liệu
    g = torch.Generator().manual_seed(cfg["project"]["seed"])  # Tạo bộ sinh số ngẫu nhiên cố định hạt giống để tái lập thứ tự trộn mẫu
    loaders = {  # Khởi tạo từ điển chứa 3 đối tượng DataLoader
        "train": DataLoader(sets["train"], batch_size=bs, shuffle=True, generator=g, num_workers=num_workers, drop_last=False),  # Bộ nạp huấn luyện: có trộn ngẫu nhiên
        "val":   DataLoader(sets["val"], batch_size=bs, shuffle=False, num_workers=num_workers),  # Bộ nạp kiểm định: giữ nguyên thứ tự mẫu
        "test":  DataLoader(sets["test"], batch_size=bs, shuffle=False, num_workers=num_workers),  # Bộ nạp kiểm thử: giữ nguyên thứ tự mẫu
    }  # Kết thúc khởi tạo từ điển bộ nạp
    return sets, loaders  # Trả về cả từ điển tập dữ liệu và từ điển bộ nạp

"""Hàm mất mát và bộ huấn luyện MLP nhận dạng Ld, Lq: tính sai số trung bình (%) ngay trong lúc huấn luyện."""  # Mô tả mô-đun
import json  # Thư viện ghi nhật ký huấn luyện ra tệp JSON
import os  # Thư viện thao tác đường dẫn tệp
import time  # Thư viện đo thời gian chạy
import torch  # Thư viện học sâu PyTorch
import torch.nn as nn  # Các lớp mạng nơ-ron và hàm mất mát của PyTorch


class RelativeMSE(nn.Module):  # Lớp định nghĩa hàm mất mát bình phương tương đối
    """Sai số bình phương trung bình của (Ld, Lq) sau khi chia cho giá trị tham chiếu."""  # Mô tả lớp

    def __init__(self, ref):  # Hàm khởi tạo hàm mất mát
        """Nhận danh sách giá trị tham chiếu (ld_ref, lq_ref) để chuẩn hóa sai số."""  # Mô tả hàm
        super().__init__()  # Gọi hàm khởi tạo của lớp cha nn.Module
        self.register_buffer("ref", torch.tensor(ref, dtype=torch.float32))  # Lưu giá trị tham chiếu vào buffer (không cập nhật gradient)

    def forward(self, pred, target):  # Hàm tính giá trị mất mát
        """Trả về trung bình bình phương sai số tương đối giữa dự đoán và nhãn."""  # Mô tả hàm
        return (((pred - target) / self.ref) ** 2).mean()  # Chia hiệu cho giá trị tham chiếu, bình phương rồi lấy trung bình


def build_loss(cfg):  # Hàm khởi tạo hàm mất mát từ tệp cấu hình
    """Dựng đối tượng RelativeMSE với tham chiếu (ld_ref, lq_ref) từ cấu hình."""  # Mô tả hàm
    m = cfg["motor"]  # Truy cập nhóm thông số động cơ trong cấu hình
    return RelativeMSE([m["ld_ref"], m["lq_ref"]])  # Khởi tạo và trả về hàm mất mát


class Trainer:  # Lớp quản lý quá trình huấn luyện mô hình
    """Huấn luyện mô hình với Adam, cắt gradient, giảm tốc độ học, dừng sớm và theo dõi sai số trung bình (%)."""  # Mô tả lớp

    def __init__(self, model, loss_fn, loaders, cfg, device, save_dir, name):  # Hàm khởi tạo bộ huấn luyện
        """Nhận mô hình, hàm mất mát, bộ nạp dữ liệu, cấu hình, thiết bị và thư mục lưu kết quả."""  # Mô tả hàm
        t = cfg["train"]  # Truy cập nhóm cấu hình huấn luyện
        self.model, self.loss_fn, self.loaders = model.to(device), loss_fn.to(device), loaders  # Chuyển mô hình, hàm mất mát lên thiết bị và lưu bộ nạp
        self.cfg, self.device, self.name = cfg, device, name  # Lưu lại cấu hình, thiết bị tính toán và tên phiên chạy
        self.opt = torch.optim.Adam(self.model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=max(1, t["patience"] // 3))  # Giảm một nửa tốc độ học khi mất mát kiểm định chững lại
        self.ckpt = os.path.join(save_dir, f"{name}_best.pt")  # Đường dẫn lưu tệp trọng số tốt nhất
        self.history = []  # Danh sách lưu lịch sử các chỉ số qua từng kỷ nguyên

    def _run_epoch(self, split):  # Phương thức chạy một kỷ nguyên trên một tập dữ liệu
        """Chạy một kỷ nguyên trên tập split, trả về (mất mát TB, sai số TB Ld %, sai số TB Lq %)."""  # Mô tả hàm
        train = split == "train"  # Xác định cờ chế độ: True nếu đang huấn luyện, False nếu kiểm định
        self.model.train(train)  # Bật chế độ train (tính gradient) hoặc eval (đánh giá) cho mô hình
        tot_loss, tot_err_ld, tot_err_lq, n = 0.0, 0.0, 0.0, 0  # Khởi tạo các biến cộng dồn mất mát, sai số và tổng số mẫu
        for batch in self.loaders[split]:  # Duyệt qua từng lô dữ liệu trong bộ nạp
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)  # Chuyển tensor đầu vào và nhãn sang CPU/GPU
            with torch.set_grad_enabled(train):  # Chỉ bật tính toán đồ thị gradient khi đang ở tập train
                pred = self.model(x)  # Lan truyền thuận qua mạng MLP để dự đoán (Ld, Lq)
                loss = self.loss_fn(pred, y)  # Tính giá trị hàm mất mát của lô hiện tại
            if train:  # Nếu đang ở giai đoạn huấn luyện
                self.opt.zero_grad()  # Xóa sạch gradient tích lũy từ bước trước
                loss.backward()  # Lan truyền ngược để tính gradient theo các trọng số
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])  # Giới hạn chuẩn gradient để tránh bùng nổ đạo hàm
                self.opt.step()  # Cập nhật trọng số mạng theo thuật toán Adam

            with torch.no_grad():  # Tắt theo dõi gradient khi tính toán chỉ số thống kê sai số
                rel_err = torch.abs(pred - y) / torch.clamp(torch.abs(y), min=1e-12) * 100.0  # Tính sai số tương đối tuyệt đối (%) cho từng mẫu
                tot_err_ld += float(rel_err[:, 0].sum())  # Cộng dồn tổng sai số phần trăm của tham số Ld trong lô
                tot_err_lq += float(rel_err[:, 1].sum())  # Cộng dồn tổng sai số phần trăm của tham số Lq trong lô

            k = x.shape[0]  # Lấy số lượng mẫu thực tế trong lô hiện tại
            tot_loss += float(loss.detach()) * k  # Cộng dồn tổng mất mát có trọng số theo kích thước lô
            n += k  # Cộng dồn tổng số mẫu đã duyệt qua
        return tot_loss / n, tot_err_ld / n, tot_err_lq / n  # Trả về giá trị trung bình trên toàn bộ tập dữ liệu

    def fit(self, print_every=10):  # Phương thức điều khiển vòng lặp huấn luyện chính
        """Huấn luyện tới khi hết số kỷ nguyên hoặc dừng sớm; in sai số trung bình (%) định kỳ."""  # Mô tả hàm
        t, best, wait, t0 = self.cfg["train"], float("inf"), 0, time.time()  # Khởi tạo tham số dừng sớm và mốc thời gian bắt đầu
        for ep in range(t["epochs"]):  # Vòng lặp duyệt qua từng kỷ nguyên huấn luyện
            tr_loss, tr_ld, tr_lq = self._run_epoch("train")  # Chạy một kỷ nguyên trên tập huấn luyện
            va_loss, va_ld, va_lq = self._run_epoch("val")  # Chạy đánh giá trên tập kiểm định
            self.sched.step(va_loss)  # Cập nhật bộ điều chỉnh tốc độ học dựa trên mất mát kiểm định

            record = {  # Đóng gói toàn bộ chỉ số của kỷ nguyên hiện tại vào từ điển
                "epoch": ep,  # Chỉ số kỷ nguyên (tính từ 0)
                "train": tr_loss,  # Mất mát trung bình tập huấn luyện
                "val": va_loss,  # Mất mát trung bình tập kiểm định
                "train_ld_err_pct": tr_ld,  # Sai số tương đối trung bình của Ld trên tập huấn luyện (%)
                "train_lq_err_pct": tr_lq,  # Sai số tương đối trung bình của Lq trên tập huấn luyện (%)
                "val_ld_err_pct": va_ld,  # Sai số tương đối trung bình của Ld trên tập kiểm định (%)
                "val_lq_err_pct": va_lq,  # Sai số tương đối trung bình của Lq trên tập kiểm định (%)
                "lr": self.opt.param_groups[0]["lr"],  # Giá trị tốc độ học hiện tại
            }  # Kết thúc đóng gói bản ghi
            self.history.append(record)  # Thêm bản ghi kỷ nguyên vào danh sách lịch sử

            if ep == 0 or (ep + 1) % print_every == 0:  # In tiến độ ở kỷ nguyên đầu tiên và định kỳ mỗi print_every kỷ nguyên
                print(
                    f"Epoch {ep+1:3d}/{t['epochs']} | "
                    f"Train Err: Ld={tr_ld:.4f}%, Lq={tr_lq:.4f}% | "
                    f"Val Err: Ld={va_ld:.4f}%, Lq={va_lq:.4f}%"
                )  # Hiển thị trực tiếp sai số trung bình (%) của Ld và Lq

            if va_loss < best:  # Nếu mất mát kiểm định đạt mức thấp kỷ lục mới
                best, wait = va_loss, 0  # Cập nhật mức tốt nhất và đặt lại bộ đếm chờ dừng sớm về 0
                self.save()  # Lưu lại trọng số mô hình tốt nhất ra tệp .pt
            else:  # Nếu mất mát kiểm định không cải thiện
                wait += 1  # Tăng bộ đếm số kỷ nguyên chờ đợi thêm 1
            if wait >= t["patience"]:  # Nếu số kỷ nguyên không cải thiện chạm ngưỡng kiên nhẫn
                print(f"Dừng sớm tại epoch {ep+1} | Val Err hiện tại: Ld={va_ld:.4f}%, Lq={va_lq:.4f}%")  # Thông báo kích hoạt dừng sớm
                break  # Thoát khỏi vòng lặp huấn luyện

        if not os.path.exists(self.ckpt):  # Kiểm tra phòng trường hợp chưa có tệp trọng số nào được lưu
            self.save()  # Lưu trạng thái mô hình cuối cùng
        return {"history": self.history, "best_val": best, "seconds": time.time() - t0}  # Trả về lịch sử, mất mát tốt nhất và tổng thời gian chạy

    def save(self):  # Phương thức lưu trọng số mô hình
        """Lưu từ điển trọng số state_dict và cấu hình kiến trúc mạng ra tệp."""  # Mô tả hàm
        torch.save({"state_dict": self.model.state_dict(), "arch": self.model.get_config()}, self.ckpt)  # Ghi tệp checkpoint bằng PyTorch

    def save_history(self, path):  # Phương thức xuất lịch sử huấn luyện
        """Ghi toàn bộ nhật ký huấn luyện qua các kỷ nguyên ra tệp JSON."""  # Mô tả hàm
        with open(path, "w", encoding="utf-8") as f:  # Mở tệp đích ở chế độ ghi mã hóa UTF-8
            json.dump(self.history, f, ensure_ascii=False, indent=1)  # Ghi dữ liệu định dạng JSON dễ đọc

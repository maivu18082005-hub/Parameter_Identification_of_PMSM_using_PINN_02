"""Kịch bản kiểm thử: MLP nhận dạng Ld, Lq theo (id, iq); Rs tính đại số từ Ld, Lq đã nhận dạng."""  # Mô tả mô-đun
import argparse  # Đọc tham số dòng lệnh
import json  # Ghi kết quả JSON
import os  # Thao tác đường dẫn
import numpy as np  # Thư viện số học mảng
import pandas as pd  # Thư viện bảng dữ liệu
import torch  # Thư viện học sâu
import matplotlib  # Thư viện vẽ đồ thị
matplotlib.use("Agg")  # Dùng nền không giao diện
import matplotlib.pyplot as plt  # Giao diện vẽ đồ thị
from src.data.dataset import load_processed, INPUT_COLS  # Hàm và hằng số dữ liệu
from src.evaluation import metrics as mt  # Các chỉ số đánh giá
from src.evaluation.metrics import AUX_COLS  # Thứ tự các cột của bộ ước lượng Rs
from src.models.ld_lq_mlp import LdLqMLP  # Mạng MLP
from src.utils.common import load_config, get_paths  # Tiện ích cấu hình

TRUE_COLS = ["Rs", "Ld", "Lq"]  # Nhãn dùng để đánh giá (Rs chỉ để đánh giá, không dùng để huấn luyện)


def load_model(ckpt_path):  # Nạp mô hình từ điểm lưu
    """Dựng lại MLP từ kiến trúc đã lưu và nạp trọng số."""  # Mô tả hàm
    ck = torch.load(ckpt_path, map_location="cpu")  # Đọc điểm lưu
    m = LdLqMLP(**ck["arch"])  # Dựng mạng từ kiến trúc đã lưu
    m.load_state_dict(ck["state_dict"])  # Nạp trọng số
    return m.eval()  # Chuyển sang chế độ suy luận


def rs_band_from_cfg(cfg):  # Dải vật lý của Rs từ cấu hình
    """Trả về (thấp, cao) = (hệ số thấp·Rs(20 °C), hệ số cao·Rs(128 °C)) theo Ω."""  # Mô tả hàm
    rc, mo = cfg["rs_estimator"], cfg["motor"]  # Nhóm cấu hình
    return mt.rs_limits(mo["rs_20c"], mo["rs_128c"], rc["band_low_factor"], rc["band_high_factor"])  # Hai biên dải


def run_test(cfg, split="test", save=True, oracle=None):  # Chạy kiểm thử
    """Đánh giá trên tập split: Ld, Lq do MLP dự đoán từ (id, iq); Rs tính đại số có lọc ba lớp từ Ld, Lq đó với Ψm = motor.psi_m cố định.
    save=False: chỉ tính sai số tham số (bỏ tái tạo dòng điện, JSON, đồ thị, đối chứng), dùng cho khảo sát trên tập kiểm định.
    oracle: tính thêm Rs từ nhãn Ld, Lq (đối chứng); mặc định bằng save."""  # Mô tả hàm
    do_oracle = save if oracle is None else bool(oracle)  # Tính Rs đối chứng từ nhãn Ld, Lq (mặc định chỉ khi save=True)
    P, ev, mo, rc = get_paths(cfg), cfg["evaluation"], cfg["motor"], cfg["rs_estimator"]  # Đường dẫn và các nhóm cấu hình
    band = rs_band_from_cfg(cfg)  # Dải vật lý của Rs
    assert "psi_m" not in rc, "Psi_m là hằng số cài đặt mô phỏng (motor.psi_m), không được ghi đè trong rs_estimator"  # Cấm hiệu chỉnh Psi_m
    psi_rs = mo["psi_m"]  # Psi_m cố định dùng cho bộ ước lượng Rs
    method, kind_f, n_sm = rc.get("method", "ratio"), rc.get("filter", "rect"), int(rc.get("l_smooth", 0))  # Dạng ước lượng, kiểu cửa sổ, độ dài làm trơn Ld/Lq
    gate_band = band if rc.get("band_gate", True) else None  # Lớp 1: loại mẫu thô ngoài dải (có thể tắt)
    clip_band = band if rc.get("clip_output", True) else None  # Lớp 3: cắt đầu ra về dải
    model = load_model(os.path.join(P["models"], "mlp_best.pt"))  # Nạp mô hình
    frames = load_processed(P["processed"], cfg["data"]["split"][split])  # Nạp các tệp của tập cần đánh giá
    preds, trues, flags, warms, auxs, infos, helds, clips, rec = [], [], [], [], [], [], [], [], {"d": [], "q": [], "md": [], "mq": []}  # Bộ tích lũy
    rs_orcs, infos_o = [], []  # Rs đối chứng từ nhãn Ld, Lq và thống kê từng lớp của nó
    for df in frames:  # Duyệt từng tệp
        x = torch.tensor(df[INPUT_COLS].to_numpy(np.float32))  # Đầu vào (id, iq)
        with torch.no_grad():  # Tắt gradient
            ldlq = model(x).numpy().astype(np.float64)  # MLP dự đoán (Ld, Lq)
        ld, lq = ldlq[:, 0].copy(), ldlq[:, 1].copy()  # Tách Ld, Lq
        if n_sm > 1:  # Làm trơn nhân quả Ld, Lq trước khi tính Rs (bỏ qua nếu l_smooth <= 1)
            ld, lq = mt.causal_ema(ld, n_sm), mt.causal_ema(lq, n_sm)  # Lọc mũ nhân quả
        a = df[AUX_COLS].to_numpy(np.float64)  # Đại lượng của bộ ước lượng Rs
        t = df["Time"].to_numpy(np.float64)  # Chuỗi thời gian
        rs, _, _, info = mt.rs_hybrid(a, ld, lq, psi_rs, rc["window"], rc["iq_min_a"], rc["causal"], fallback=mo["rs_20c"], band=gate_band, min_valid_ratio=rc["min_valid_ratio"], clip=clip_band, return_info=True, method=method, t=t, kind=kind_f)  # Rs tính từ Ld, Lq của MLP
        p = np.column_stack([rs, ld, lq])  # Ghép (Rs, Ld, Lq)
        helds.append(info.pop("held_mask")), clips.append(info.pop("clipped_mask"))  # Mặt nạ giữ giá trị và cắt, không đưa vào thống kê trung bình
        infos.append(info)  # Thống kê từng lớp của tệp
        if do_oracle:  # Rs đối chứng, cùng bộ ước lượng và cấu hình, chỉ thay Ld, Lq của MLP bằng nhãn
            lab_f = df[TRUE_COLS].to_numpy(np.float64)  # Nhãn (Rs, Ld, Lq)
            r_o, _, _, info_o = mt.rs_oracle(a, lab_f[:, 1], lab_f[:, 2], psi_rs, rc, band, mo["rs_20c"], t)  # Rs từ nhãn Ld, Lq
            info_o.pop("held_mask"), info_o.pop("clipped_mask")  # Mặt nạ không đưa vào thống kê trung bình
            rs_orcs.append(r_o), infos_o.append(info_o)  # Lưu Rs đối chứng và thống kê
        if save:  # Tái tạo dòng điện chỉ khi cần lưu kết quả đầy đủ
            dt = float(np.median(np.diff(t)))  # Chu kỳ giữa hai mẫu
            n_h = max(2, int(round(ev["recon_horizon_s"] / dt)))  # Số bước mỗi đoạn mô phỏng
            rd, rq, md, mq = mt.reconstruct_currents(p, a, dt, n_h, mo["psi_m"])  # Tái tạo dòng điện
            for k, v in zip(("d", "q", "md", "mq"), (rd, rq, md, mq)):  # Duyệt bốn mảng
                rec[k].append(v.ravel())  # Làm phẳng và lưu
        preds.append(p), trues.append(df[TRUE_COLS].to_numpy(np.float64))  # Lưu dự đoán và nhãn
        flags.append(df["is_transient"].to_numpy(bool)), auxs.append(a)  # Lưu cờ quá độ và đại lượng phụ
        warms.append(np.arange(len(p)) < int(rc["window"]))  # Cờ đoạn khởi động: cửa sổ lọc chưa đầy
    pred, true, flg, warm = np.vstack(preds), np.vstack(trues), np.concatenate(flags), np.concatenate(warms)  # Gộp các tệp
    masks = mt.condition_masks(flg)  # Mặt nạ điều kiện
    rows = mt.evaluate_parameters(pred, true, masks, ev["tolerances"], ev["verdict_stat"])  # Bảng sai số
    out = {"model": "mlp", "split": split, "parameters": rows, "current_rmse": None}  # Gói kết quả
    masks_nw = {k: m & ~warm for k, m in masks.items()}  # Mặt nạ loại đoạn khởi động
    out["parameters_excl_startup"] = mt.evaluate_parameters(pred, true, masks_nw, ev["tolerances"], ev["verdict_stat"])  # Sai số khi loại đoạn khởi động
    sel = masks_nw["Rs_steady"]  # Mẫu Rs xác lập sau khởi động
    out["rs_signed_bias_pct"] = float(np.mean((pred[sel, 0] - true[sel, 0]) / true[sel, 0]) * 100.0) if sel.any() else float("nan")  # Sai lệch Rs trung bình có dấu (%)
    out["rs_estimator"] = {"psi_m": psi_rs, "method": method, "filter": kind_f, "l_smooth": n_sm, "window": rc["window"], "iq_min_a": rc["iq_min_a"], "causal": rc["causal"], "band": list(band), "band_gate": gate_band is not None, "clip_output": clip_band is not None, "min_valid_ratio": rc["min_valid_ratio"], **{k: float(np.mean([i[k] for i in infos])) for k in infos[0]}}  # Cấu hình và thống kê từng lớp
    err_rs = mt.relative_error_pct(pred[:, 0], true[:, 0])  # Sai số tương đối Rs từng mẫu (%)
    edges = np.linspace(0, len(pred), 11).astype(int)  # Biên mười đoạn liên tiếp
    out["rs_segments"] = [{"seg": k, **mt.error_stats(err_rs[lo:hi][sel[lo:hi]])} for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]  # Sai số Rs xác lập theo mười đoạn
    top = np.flatnonzero(sel)[np.argsort(err_rs[sel])[::-1][:8]]  # Tám mẫu Rs xác lập sai nhiều nhất, sau khởi động
    held_all, clip_all, iq_all = np.concatenate(helds), np.concatenate(clips), np.vstack(auxs)[:, 3]  # Cờ giữ, cờ cắt, dòng iq
    out["rs_worst"] = [{"idx": int(i), "rs_true": float(true[i, 0]), "rs_pred": float(pred[i, 0]), "err_pct": float(err_rs[i]), "iq": float(iq_all[i]), "held": bool(held_all[i]), "clipped": bool(clip_all[i])} for i in top]  # Ngữ cảnh các mẫu sai lớn
    if do_oracle:  # Kết quả đối chứng, cùng mặt nạ và cùng cách tính như đường chính
        rs_or = np.concatenate(rs_orcs)  # Rs từ nhãn Ld, Lq, gộp các tệp
        err_or = mt.relative_error_pct(rs_or, true[:, 0])  # Sai số tương đối Rs đối chứng (%)
        lim_rs, trn = ev["tolerances"]["Rs"], masks_nw["Rs_transient"]  # Dung sai Rs và mặt nạ quá độ sau khởi động
        rows_o = []  # Bảng kết quả đối chứng
        for cond, m_ in (("steady", sel), ("transient", trn)):  # Xác lập và quá độ
            s_o = mt.error_stats(err_or[m_])  # Thống kê sai số
            rows_o.append({"param": "Rs", "condition": cond, "limit_pct": lim_rs[cond], **s_o, "pass": bool(s_o["n"] > 0 and s_o[ev["verdict_stat"]] <= lim_rs[cond])})  # Một hàng
        out["rs_oracle"] = {"source": "Rs từ nhãn Ld, Lq, cùng bộ ước lượng và cấu hình", "parameters_excl_startup": rows_o, "signed_bias_pct": float(np.mean((rs_or[sel] - true[sel, 0]) / true[sel, 0]) * 100.0) if sel.any() else float("nan"), "frac_held": float(np.mean([i["frac_held"] for i in infos_o])), "frac_clipped": float(np.mean([i["frac_clipped"] for i in infos_o])), "frac_mlp_better_than_oracle": float(np.mean(err_rs[sel] < err_or[sel])) if sel.any() else float("nan"), "segments": [{"seg": k, **mt.error_stats(err_or[lo:hi][sel[lo:hi]])} for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]}  # Thống kê đối chứng, tỷ lệ mẫu MLP tốt hơn đối chứng và sai số theo mười đoạn
    if not save:  # Khảo sát nhanh: không ghi tệp
        return out  # Trả về kết quả
    cat = {k: np.concatenate(v) for k, v in rec.items()}  # Gộp dòng điện các tệp
    rmse = {"id_pct": mt.current_rmse_pct(cat["d"], cat["md"], mo["i_rated_peak"]), "iq_pct": mt.current_rmse_pct(cat["q"], cat["mq"], mo["i_rated_peak"])}  # RMSE dòng điện
    rmse["pass"] = bool(max(rmse["id_pct"], rmse["iq_pct"]) <= ev["rmse_limit_pct"])  # Kết luận RMSE
    out["current_rmse"] = rmse  # Ghi RMSE vào kết quả
    os.makedirs(P["results"], exist_ok=True)  # Tạo thư mục kết quả
    with open(os.path.join(P["results"], f"test_mlp_rs_{split}.json"), "w", encoding="utf-8") as f:  # Mở tệp JSON
        json.dump(out, f, ensure_ascii=False, indent=1)  # Ghi kết quả
    plot_tracking(pred, true, os.path.join(P["results"], f"tracking_mlp_rs_{split}.png"), rs_band=band, tol_pct=ev["tolerances"]["Rs"]["steady"], rs_oracle=np.concatenate(rs_orcs) if do_oracle else None)  # Vẽ đồ thị bám nhãn, kèm Rs đối chứng nếu có
    return out  # Trả về kết quả


def plot_tracking(pred, true, path, max_points=4000, rs_band=None, tol_pct=3.0, rs_oracle=None):  # Vẽ đồ thị so sánh
    """Vẽ dự đoán so với nhãn của Rs, Ld, Lq theo chỉ số mẫu; Rs có thêm dải vật lý, đường đối chứng và đồ thị sai số lớn nhất mỗi khoảng."""  # Mô tả hàm
    idx = np.linspace(0, len(pred) - 1, min(max_points, len(pred))).astype(int)  # Chỉ số lấy mẫu con
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)  # Bốn đồ thị chồng dọc
    for j, (name, unit) in enumerate((("Rs", "Ω"), ("Ld", "H"), ("Lq", "H"))):  # Duyệt ba tham số
        ax[j].plot(idx, true[idx, j], label="Nhãn")  # Đường nhãn
        ax[j].plot(idx, pred[idx, j], label="Rs đại số từ Ld, Lq của MLP" if j == 0 else "MLP", linewidth=0.8)  # Đường dự đoán
        if j == 0 and rs_oracle is not None:  # Rs: thêm đường Rs tính từ nhãn Ld, Lq (đối chứng)
            ax[j].plot(idx, rs_oracle[idx], label="Rs từ nhãn Ld, Lq", color="tab:red", linestyle="--", linewidth=0.9)  # Đường Rs đối chứng
        if j == 0 and rs_band is not None:  # Rs: vẽ hai biên dải vật lý
            ax[j].axhline(rs_band[0], color="gray", linestyle="--", linewidth=0.6, label="Dải vật lý")  # Biên dưới
            ax[j].axhline(rs_band[1], color="gray", linestyle="--", linewidth=0.6)  # Biên trên
        ax[j].set_ylabel(f"{name} ({unit})")  # Nhãn trục tung
        ax[j].legend()  # Chú giải
    err = np.abs(pred[:, 0] - true[:, 0]) / np.maximum(np.abs(true[:, 0]), 1e-12) * 100.0  # Sai số tương đối Rs từng mẫu (%)
    ax[3].plot(idx, np.maximum.reduceat(err, idx), linewidth=0.6, label="Sai số Rs lớn nhất mỗi khoảng (Ld, Lq của MLP)")  # Cực đại trong từng khoảng, không bỏ sót xung ngắn
    if rs_oracle is not None:  # Sai số của Rs đối chứng
        err_o = np.abs(rs_oracle - true[:, 0]) / np.maximum(np.abs(true[:, 0]), 1e-12) * 100.0  # Sai số tương đối Rs đối chứng (%)
        ax[3].plot(idx, np.maximum.reduceat(err_o, idx), linewidth=0.6, color="tab:red", label="Sai số Rs lớn nhất mỗi khoảng (Rs từ nhãn L)")  # Cực đại trong từng khoảng
    ax[3].axhline(tol_pct, color="gray", linestyle="--", linewidth=0.6, label=f"Ngưỡng {tol_pct:g}%")  # Ngưỡng xác lập
    ax[3].set_ylabel("Sai số Rs (%)")  # Nhãn trục tung
    ax[3].legend()  # Chú giải
    ax[-1].set_xlabel("Chỉ số mẫu")  # Nhãn trục hoành
    fig.tight_layout()  # Căn chỉnh bố cục
    fig.savefig(path, dpi=150)  # Lưu ảnh
    plt.close(fig)  # Giải phóng bộ nhớ


def main():  # Hàm dòng lệnh
    """Chạy kiểm thử từ dòng lệnh: python -m src.evaluation.test --split test."""  # Mô tả hàm
    ap = argparse.ArgumentParser()  # Bộ phân tích tham số
    ap.add_argument("--config", default="configs/config.yaml")  # Đường dẫn cấu hình
    ap.add_argument("--split", default="test")  # Tập đánh giá
    a = ap.parse_args()  # Đọc tham số
    res = run_test(load_config(a.config), a.split)  # Chạy kiểm thử
    print(json.dumps(res["current_rmse"], ensure_ascii=False))  # In RMSE dòng điện


if __name__ == "__main__":  # Điểm vào khi chạy trực tiếp
    main()  # Gọi hàm chính

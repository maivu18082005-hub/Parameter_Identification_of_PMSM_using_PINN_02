"""Chỉ số đánh giá: sai số tương đối từng tham số theo điều kiện, RMSE dòng điện tái tạo và bộ ước lượng Rs đại số có lọc."""  # Mô tả mô-đun
import numpy as np  # Thư viện số học mảng
from scipy.signal import lfilter, lfilter_zi  # Bộ lọc IIR nhân quả

AUX_COLS = ["ud", "uq", "Id", "Iq", "omega_e", "did_dt", "diq_dt"]  # Đại lượng của bộ ước lượng Rs và tái tạo dòng; thứ tự các cột của mảng aux


def relative_error_pct(pred, true):  # Sai số tương đối
    """Trả về sai số tương đối tuyệt đối theo phần trăm."""  # Mô tả hàm
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-12) * 100.0  # |ŷ−y|/|y|·100


def error_stats(err):  # Thống kê sai số
    """Trả về trung bình, RMS, P95, P99 và cực đại của mảng sai số."""  # Mô tả hàm
    if len(err) == 0:  # Không có mẫu
        return {"n": 0, "mean": np.nan, "rms": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}  # Trả về rỗng
    return {"n": int(len(err)), "mean": float(err.mean()), "rms": float(np.sqrt((err ** 2).mean())), "p95": float(np.percentile(err, 95)), "p99": float(np.percentile(err, 99)), "max": float(err.max())}  # Các thống kê


def condition_masks(is_transient):  # Mặt nạ điều kiện đánh giá
    """Dựng mặt nạ: Rs tách xác lập và quá độ; Ld, Lq đánh giá trên toàn bộ mẫu."""  # Mô tả hàm
    trans = np.asarray(is_transient, dtype=bool)  # Cờ quá độ
    return {  # Từ điển mặt nạ
        "Rs_steady": ~trans, "Rs_transient": trans,  # Điều kiện của Rs
        "Ld_all": np.ones_like(trans), "Lq_all": np.ones_like(trans),  # Ld, Lq: mọi mẫu
    }  # Kết thúc từ điển


def evaluate_parameters(pred, true, masks, tolerances, verdict_stat="max"):  # Đánh giá tham số
    """Tính thống kê sai số và kết luận đạt/không đạt cho từng tham số và điều kiện."""  # Mô tả hàm
    rows = []  # Danh sách kết quả
    for j, p in enumerate(("Rs", "Ld", "Lq")):  # Duyệt ba tham số
        err = relative_error_pct(pred[:, j], true[:, j])  # Sai số tương đối từng mẫu
        for cond, lim in tolerances[p].items():  # Duyệt từng điều kiện
            s = error_stats(err[masks[f"{p}_{cond}"]])  # Thống kê trong điều kiện
            ok = bool(s["n"] > 0 and s[verdict_stat] <= lim)  # Kết luận theo thống kê đã chọn
            rows.append({"param": p, "condition": cond, "limit_pct": lim, **s, "pass": ok})  # Ghi một hàng
    return rows  # Trả về bảng kết quả


def reconstruct_currents(pred, aux, dt, horizon_steps, psi_m):  # Tái tạo dòng điện
    """Mô phỏng id, iq theo mô hình điện với tham số nhận dạng và Psi_m cố định trên các đoạn độc lập."""  # Mô tả hàm
    rs, ld, lq = pred[:, 0], pred[:, 1], pred[:, 2]  # Tách tham số
    ud, uq, i_d, i_q, w = [aux[:, k] for k in range(5)]  # Tách đại lượng đo
    n_seg = len(rs) // horizon_steps  # Số đoạn đầy đủ
    cut = n_seg * horizon_steps  # Số mẫu dùng
    rs, ld, lq, ud, uq, i_d, i_q, w = [a[:cut].reshape(n_seg, horizon_steps) for a in (rs, ld, lq, ud, uq, i_d, i_q, w)]  # Chia thành các đoạn
    sd, sq = i_d[:, 0].copy(), i_q[:, 0].copy()  # Điều kiện đầu lấy từ giá trị đo
    out_d, out_q = np.zeros_like(i_d), np.zeros_like(i_q)  # Mảng kết quả
    for k in range(horizon_steps):  # Duyệt từng bước thời gian
        out_d[:, k], out_q[:, k] = sd, sq  # Ghi trạng thái hiện tại
        dd = (ud[:, k] - rs[:, k] * sd + w[:, k] * lq[:, k] * sq) / ld[:, k]  # Đạo hàm id theo mô hình
        dq = (uq[:, k] - rs[:, k] * sq - w[:, k] * ld[:, k] * sd - w[:, k] * psi_m) / lq[:, k]  # Đạo hàm iq theo mô hình
        sd, sq = sd + dt * dd, sq + dt * dq  # Tích phân Euler
    return out_d, out_q, i_d, i_q  # Trả về dòng tái tạo và dòng đo



def current_rmse_pct(rec, meas, i_base):  # RMSE dòng điện
    """Trả về RMSE chuẩn hóa theo dòng cơ sở (%)."""  # Mô tả hàm
    return float(np.sqrt(np.mean((rec - meas) ** 2)) / i_base * 100.0)  # RMSE chia dòng cơ sở


def rs_limits(rs_20c, rs_128c, low_factor=0.9, high_factor=1.1):  # Dải vật lý của Rs
    """Trả về (thấp, cao) = (low_factor·Rs(20 °C), high_factor·Rs(128 °C)) theo Ω."""  # Mô tả hàm
    return float(low_factor * rs_20c), float(high_factor * rs_128c)  # Hai biên của dải vật lý


def rs_numerator(aux, ld, lq, psi_m):  # Tử số của Rs theo phương trình trục q
    """Trả về (num, iq) với num = uq − Lq·diq/dt − ωe·(Ld·id + Ψm), sao cho Rs = num/iq."""  # Mô tả hàm
    uq, i_d, i_q, w, diq = aux[:, 1], aux[:, 2], aux[:, 3], aux[:, 4], aux[:, 6]  # Tách đại lượng đo (thứ tự AUX_COLS)
    return uq - lq * diq - w * (ld * i_d + psi_m), i_q  # Tử số và dòng iq


def rs_algebraic(aux, ld, lq, psi_m, iq_min, band=None):  # Rs thô từ phương trình trục q
    """Rs thô (Ω) = num/iq. Loại mẫu |iq| < iq_min (lớp ngưỡng dòng) và, nếu có band, mẫu Rs thô ngoài [thấp, cao] (lớp 1)."""  # Mô tả hàm
    num, i_q = rs_numerator(aux, ld, lq, psi_m)  # Tử số và iq
    valid = np.abs(i_q) >= iq_min  # Chỉ giữ mẫu có dòng iq đủ lớn
    safe_iq = np.where(valid, i_q, 1.0)  # Tránh chia cho số quá nhỏ ở mẫu bị loại
    rs = num / safe_iq  # Giải phương trình trục q theo Rs
    valid = valid & np.isfinite(rs)  # Loại giá trị không hữu hạn
    if band is not None:  # Lớp 1: loại mẫu thô ngoài dải vật lý
        valid = valid & (rs >= band[0]) & (rs <= band[1])  # Giữ mẫu trong dải
    return np.where(valid, rs, np.nan), valid  # Rs thô (NaN ở mẫu bị loại) và cờ hợp lệ


def masked_moving_average(values, valid, window, causal=True):  # Trung bình trượt bỏ qua mẫu không hợp lệ
    """Trung bình trượt cửa sổ window mẫu, chỉ tính các mẫu hợp lệ; causal=True chỉ dùng mẫu quá khứ."""  # Mô tả hàm
    n = len(values)  # Số mẫu
    v = np.where(valid, values, 0.0)  # Giá trị, mẫu không hợp lệ bằng 0
    cs_v = np.concatenate([[0.0], np.cumsum(v)])  # Tổng tích lũy giá trị
    cs_m = np.concatenate([[0.0], np.cumsum(valid.astype(np.float64))])  # Tổng tích lũy số mẫu hợp lệ
    idx = np.arange(n)  # Chỉ số mẫu
    if causal:  # Cửa sổ nhân quả: [i - window + 1, i]
        lo, hi = np.maximum(idx - window + 1, 0), idx + 1  # Biên cửa sổ
    else:  # Cửa sổ đối xứng quanh mẫu hiện tại
        lo, hi = np.maximum(idx - window // 2, 0), np.minimum(idx + window // 2 + 1, n)  # Biên cửa sổ
    s, c = cs_v[hi] - cs_v[lo], cs_m[hi] - cs_m[lo]  # Tổng và số mẫu hợp lệ trong cửa sổ
    out = np.full(n, np.nan)  # Kết quả ban đầu là NaN
    ok = c > 0  # Cửa sổ có ít nhất một mẫu hợp lệ
    out[ok] = s[ok] / c[ok]  # Trung bình trên các mẫu hợp lệ
    return out  # Trả về chuỗi đã lọc


def masked_ratio_filter(num, iq, valid, window, min_count=1, causal=True):  # Lọc theo tỷ số các tổng
    """Rs = Σ(num·iq)/Σ(iq²) trên các mẫu hợp lệ trong cửa sổ; NaN nếu có ít hơn min_count mẫu hợp lệ (lớp 2)."""  # Mô tả hàm
    n = len(num)  # Số mẫu
    a = np.where(valid, num * iq, 0.0)  # Số hạng num·iq, mẫu không hợp lệ bằng 0
    b = np.where(valid, iq * iq, 0.0)  # Số hạng iq², mẫu không hợp lệ bằng 0
    cs_a = np.concatenate([[0.0], np.cumsum(a)])  # Tổng tích lũy num·iq
    cs_b = np.concatenate([[0.0], np.cumsum(b)])  # Tổng tích lũy iq²
    cs_m = np.concatenate([[0.0], np.cumsum(valid.astype(np.float64))])  # Tổng tích lũy số mẫu hợp lệ
    idx = np.arange(n)  # Chỉ số mẫu
    if causal:  # Cửa sổ nhân quả: [i - window + 1, i]
        lo, hi = np.maximum(idx - window + 1, 0), idx + 1  # Biên cửa sổ
    else:  # Cửa sổ đối xứng quanh mẫu hiện tại
        lo, hi = np.maximum(idx - window // 2, 0), np.minimum(idx + window // 2 + 1, n)  # Biên cửa sổ
    sa, sb, c = cs_a[hi] - cs_a[lo], cs_b[hi] - cs_b[lo], cs_m[hi] - cs_m[lo]  # Các tổng trong cửa sổ
    out = np.full(n, np.nan)  # Kết quả ban đầu là NaN
    ok = (c >= max(int(min_count), 1)) & (sb > 0)  # Đủ số mẫu hợp lệ tối thiểu
    out[ok] = sa[ok] / sb[ok]  # Tỷ số các tổng
    return out  # Trả về chuỗi đã lọc


def hold_last(x, fallback):  # Giữ giá trị hợp lệ gần nhất
    """Thay NaN bằng giá trị hợp lệ gần nhất phía trước; trước giá trị hợp lệ đầu tiên dùng fallback."""  # Mô tả hàm
    n = len(x)  # Số mẫu
    idx = np.where(np.isfinite(x), np.arange(n), -1)  # Chỉ số mẫu hợp lệ, -1 nếu chưa có
    idx = np.maximum.accumulate(idx)  # Chỉ số mẫu hợp lệ gần nhất phía trước
    return np.where(idx >= 0, x[np.maximum(idx, 0)], fallback)  # Giữ giá trị, hoặc dùng fallback


def step_sizes(t):  # Bước thời gian giữa các mẫu
    """Trả về dt[k] = t[k] − t[k−1]; mẫu đầu và các bước không dương dùng trung vị của các bước dương."""  # Mô tả hàm
    t = np.asarray(t, dtype=np.float64)  # Chuỗi thời gian
    dt = np.empty(len(t))  # Mảng bước thời gian
    dt[1:] = np.diff(t)  # Bước giữa hai mẫu liên tiếp
    pos = dt[1:][dt[1:] > 0]  # Các bước dương
    med = float(np.median(pos)) if len(pos) else 1.0  # Bước đại diện
    dt[0] = med  # Mẫu đầu dùng bước đại diện
    return np.where(dt > 0, dt, med)  # Thay bước không dương bằng bước đại diện


def rs_integral_terms(aux, t, ld, lq, psi_m):  # Số hạng dạng tích phân của phương trình trục q
    """Trả về (a, b, iq) với a = (uq − ωe·(Ld·id + Ψm))·dt − Lq·Δiq và b = iq·dt, sao cho Rs ≈ Σa/Σb trên một cửa sổ.
    Δiq là hiệu lùi iq[k] − iq[k−1] (nhân quả); tổng Lq·Δiq trên cửa sổ triệt tiêu dần (telescoping) nên không cần đạo hàm đã làm trơn."""  # Mô tả hàm
    uq, i_d, i_q, w = aux[:, 1], aux[:, 2], aux[:, 3], aux[:, 4]  # Tách đại lượng đo (thứ tự AUX_COLS)
    dt = step_sizes(t)  # Bước thời gian từng mẫu
    di = np.zeros(len(i_q))  # Hiệu lùi của iq
    di[1:] = np.diff(i_q)  # Δiq[k] = iq[k] − iq[k−1]
    a = (uq - w * (ld * i_d + psi_m)) * dt - lq * di  # Tử số tích phân
    return a, i_q * dt, i_q  # Tử số, mẫu số, dòng iq


def window_sum(x, window, kind="rect", causal=True):  # Tổng trượt
    """Tổng trượt của x. kind='rect': cửa sổ chữ nhật window mẫu; kind='ema': tổng trọng số mũ với λ = 1 − 1/window (chỉ nhân quả)."""  # Mô tả hàm
    x = np.asarray(x, dtype=np.float64)  # Chuỗi đầu vào
    n = len(x)  # Số mẫu
    if kind == "ema":  # Tổng trọng số mũ
        if not causal:  # EMA chỉ có dạng nhân quả
            raise ValueError("kind='ema' chỉ hỗ trợ causal=True")  # Báo lỗi cấu hình
        return lfilter([1.0], [1.0, -(1.0 - 1.0 / float(window))], x)  # Bộ lọc IIR bậc một
    cs = np.concatenate([[0.0], np.cumsum(x)])  # Tổng tích lũy
    idx = np.arange(n)  # Chỉ số mẫu
    if causal:  # Cửa sổ nhân quả: [i − window + 1, i]
        lo, hi = np.maximum(idx - window + 1, 0), idx + 1  # Biên cửa sổ
    else:  # Cửa sổ đối xứng
        lo, hi = np.maximum(idx - window // 2, 0), np.minimum(idx + window // 2 + 1, n)  # Biên cửa sổ
    return cs[hi] - cs[lo]  # Tổng trong cửa sổ


def masked_sum_ratio(a, b, valid, window, min_count=1, kind="rect", causal=True, guard=0.5):  # Tỷ số Σa/Σb có mặt nạ
    """Σ(a·valid)/Σ(b·valid) trên cửa sổ trượt. NaN nếu tổng trọng số mẫu hợp lệ < min_count (lớp 2) hoặc |Σb| < guard·Σ|b|
    (iq đổi dấu trong cửa sổ làm mẫu số gần triệt tiêu)."""  # Mô tả hàm
    a, b, valid = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(valid, bool)  # Chuẩn hóa kiểu
    sa = window_sum(np.where(valid, a, 0.0), window, kind, causal)  # Tổng tử số
    sb = window_sum(np.where(valid, b, 0.0), window, kind, causal)  # Tổng mẫu số
    sab = window_sum(np.where(valid, np.abs(b), 0.0), window, kind, causal)  # Tổng |mẫu số|
    c = window_sum(valid.astype(np.float64), window, kind, causal)  # Tổng trọng số mẫu hợp lệ
    ok = (c >= max(float(min_count), 1.0)) & (sab > 0) & (np.abs(sb) >= guard * sab)  # Điều kiện cập nhật
    out = np.full(len(a), np.nan)  # Kết quả ban đầu là NaN
    out[ok] = sa[ok] / sb[ok]  # Tỷ số các tổng
    return out  # Trả về chuỗi đã lọc


def causal_ema(x, n):  # Làm trơn mũ nhân quả
    """Lọc IIR bậc một nhân quả y[k] = y[k−1] + (x[k] − y[k−1])/n, khởi tạo bằng x[0]; n ≤ 1 trả về bản sao của x."""  # Mô tả hàm
    x = np.asarray(x, dtype=np.float64)  # Chuỗi đầu vào
    if n is None or n <= 1:  # Không làm trơn
        return x.copy()  # Trả về bản sao
    b, a = [1.0 / n], [1.0, -(1.0 - 1.0 / n)]  # Hệ số bộ lọc
    y, _ = lfilter(b, a, x, zi=lfilter_zi(b, a) * x[0])  # Lọc với trạng thái đầu bằng x[0]
    return y  # Chuỗi đã làm trơn


def rs_hybrid(aux, ld, lq, psi_m, window, iq_min, causal=True, fallback=None, band=None, min_valid_ratio=0.0, clip=None, return_info=False, method="ratio", t=None, kind="rect"):  # Rs đại số có lọc
    """Rs đại số từ Ld, Lq đã nhận dạng và Ψm cố định (hằng số cài đặt mô phỏng, không hiệu chỉnh).
    method='ratio': Σ(num·iq)/Σ(iq²) với diq_dt đã làm trơn (cột 6 của aux); method='integral': Σa/Σb dạng tích phân dùng hiệu lùi của iq và thời gian t.
    kind='rect' (cửa sổ chữ nhật) hoặc 'ema' (trọng số mũ, nhân quả). Lớp 1: loại mẫu thô ngoài band; lớp 2: chỉ cập nhật khi tổng trọng số
    mẫu hợp lệ ≥ ceil(min_valid_ratio·window), nếu không giữ giá trị trước (đầu tệp dùng fallback); lớp 3: cắt về clip.
    Trả về (Rs, Rs thô, cờ hợp lệ), kèm từ điển thống kê (và mặt nạ held_mask, clipped_mask) nếu return_info=True."""  # Mô tả hàm
    assert method in ("ratio", "integral") and kind in ("rect", "ema"), "method hoặc kind không hợp lệ"  # Kiểm tra tham số
    num, i_q = rs_numerator(aux, ld, lq, psi_m)  # Tử số và iq
    raw, v_iq = rs_algebraic(aux, ld, lq, psi_m, iq_min)  # Rs thô, chỉ lọc theo ngưỡng |iq|
    valid = v_iq if band is None else (v_iq & (raw >= band[0]) & (raw <= band[1]))  # Lớp 1: giữ mẫu trong dải vật lý
    min_count = max(1, int(np.ceil(float(min_valid_ratio) * int(window))))  # Số mẫu hợp lệ tối thiểu của cửa sổ
    if method == "integral":  # Dạng tích phân: không dùng đạo hàm đã làm trơn
        assert t is not None, "method='integral' cần chuỗi thời gian t"  # Cần thời gian để tính dt
        a_i, b_i, _ = rs_integral_terms(aux, t, ld, lq, psi_m)  # Số hạng tích phân
        filt = masked_sum_ratio(a_i, b_i, valid, int(window), min_count, kind, causal)  # Lớp 2: tỷ số các tổng
    elif kind == "rect":  # Dạng cũ của v4
        filt = masked_ratio_filter(num, i_q, valid, int(window), min_count, causal)  # Lớp 2: lọc tỷ số các tổng, NaN nếu thiếu mẫu
    else:  # Tỷ số Σ(num·iq)/Σ(iq²) với trọng số mũ
        filt = masked_sum_ratio(num * i_q, i_q * i_q, valid, int(window), min_count, kind, causal)  # Lớp 2
    n = len(filt)  # Số mẫu
    fb = np.full(n, np.nan) if fallback is None else np.broadcast_to(np.asarray(fallback, dtype=np.float64), (n,))  # Giá trị khởi động
    out = hold_last(filt, fb)  # Giữ giá trị trước khi chưa đủ mẫu hợp lệ
    held = ~np.isfinite(filt)  # Mẫu đang giữ giá trị hoặc dùng giá trị khởi động
    clipped = np.zeros(n, dtype=bool)  # Cờ mẫu bị cắt
    if clip is not None:  # Lớp 3: cắt đầu ra về dải vật lý
        cut = np.clip(out, clip[0], clip[1])  # Đầu ra sau khi cắt
        clipped = np.isfinite(out) & (cut != out)  # Mẫu bị cắt
        out = cut  # Dùng đầu ra đã cắt
    raw_used = np.where(valid, raw, np.nan)  # Rs thô của mẫu hợp lệ cuối cùng
    if not return_info:  # Dạng trả về cơ bản
        return out, raw_used, valid  # Rs, Rs thô, cờ hợp lệ
    info = {"frac_iq_ok": float(v_iq.mean()), "frac_used": float(valid.mean()), "frac_band_rejected": float((v_iq & ~valid).mean()), "frac_held": float(held.mean()), "frac_clipped": float(clipped.mean()), "held_mask": held, "clipped_mask": clipped}  # Thống kê từng lớp và mặt nạ
    return out, raw_used, valid, info  # Rs, Rs thô, cờ hợp lệ, thống kê


def rs_oracle(aux, ld_lab, lq_lab, psi_m, rc, band, fallback, t):  # Rs đối chứng từ nhãn Ld, Lq
    """Rs đại số có lọc (cùng bộ ước lượng, cùng cấu hình rs_estimator như đường chính) nhưng dùng NHÃN Ld, Lq thay cho Ld, Lq của MLP.
    Không làm trơn Ld, Lq (nhãn đã là đáp án). Trả về (Rs, Rs thô, cờ hợp lệ, thống kê kèm held_mask và clipped_mask) như rs_hybrid với return_info=True."""  # Mô tả hàm
    gate = band if rc.get("band_gate", True) else None  # Lớp 1 theo cấu hình
    clip = band if rc.get("clip_output", True) else None  # Lớp 3 theo cấu hình
    return rs_hybrid(aux, ld_lab, lq_lab, psi_m, int(rc["window"]), rc["iq_min_a"], rc["causal"], fallback=fallback, band=gate, min_valid_ratio=rc["min_valid_ratio"], clip=clip, return_info=True, method=rc.get("method", "ratio"), t=t, kind=rc.get("filter", "rect"))  # Cùng cấu hình với đường chính

"""
Giao diện phân tích cảm xúc theo khía cạnh (ABSA) cho review tiếng Việt.
Gọi Backend FastAPI + PhoBERT theo đúng file API_CONTRACT.md.

Chạy:  streamlit run app.py
(Backend phải đang chạy ở http://127.0.0.1:8000)

Ứng dụng có 2 tab:
  1. Phân tích: gọi POST /api/v1/analyze (không lưu) hoặc POST /api/v1/reviews (lưu)
  2. Dashboard: gọi GET /api/v1/reviews và GET /api/v1/statistics với cùng bộ lọc

Giao diện KHÔNG tự lưu dữ liệu. Việc lưu và thống kê do Backend làm.

Bản bàn giao: chỉ gọi Backend thật, không tích hợp dữ liệu giả.
Lưu ý: đường dẫn và JSON fields cần được xác nhận với Backend FastAPI thực tế.
"""
import html
from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ====================== CẤU HÌNH ======================
BASE_URL_MAC_DINH = "http://127.0.0.1:8000"

MAU_CHU = {"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#d97706"}
MAU_NEN = {"Positive": "#bbf7d0", "Negative": "#fecaca", "Neutral": "#fde68a"}
TEN_VN = {"Positive": "Tích cực", "Negative": "Tiêu cực", "Neutral": "Trung tính"}
KHONG_RO = "Không xác định"

# Màu giao diện: nền chính trắng, các khung (card) xám nhạt để tách biệt từng phần
NEN_CHINH = "#ffffff"
NEN_KHUNG = "#f4f6fa"
VIEN_KHUNG = "#dfe4ec"
CHU_CHINH = "#0f172a"

st.set_page_config(page_title="Phân tích review", layout="wide")

st.markdown(f"""
<style>
/* Nền chính màu trắng */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background: {NEN_CHINH};
    color: {CHU_CHINH};
}}
/* Thanh bên hơi xám để tách khỏi phần nội dung */
[data-testid="stSidebar"] {{
    background: {NEN_KHUNG};
    border-right: 1px solid {VIEN_KHUNG};
}}
/* Khung (card): mọi container có key bắt đầu bằng "khung" */
[class*="st-key-khung"] {{
    background: {NEN_KHUNG};
    border: 1px solid {VIEN_KHUNG};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, .06), 0 1px 2px rgba(15, 23, 42, .04);
}}
/* Ô nhập liệu nằm trong khung xám thì để nền trắng cho nổi */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div,
[data-baseweb="base-input"], .stTextArea textarea, .stTextInput input, .stDateInput input {{
    background: #ffffff !important;
    color: {CHU_CHINH} !important;
}}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{
    border-color: #cbd5e1 !important;
}}
/* Ô số liệu: thẻ trắng nằm trong khung xám */
[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid {VIEN_KHUNG};
    border-radius: 10px;
    padding: 10px 14px;
}}
/* Bảng: viền bo góc cho tách khỏi nền khung */
[data-testid="stDataFrame"] {{
    border: 1px solid {VIEN_KHUNG};
    border-radius: 8px;
    overflow: hidden;
}}
/* Tab: gạch chân rõ hơn trên nền trắng */
.stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {VIEN_KHUNG}; }}
/* Điện thoại: thu gọn khoảng đệm và cỡ chữ */
@media (max-width: 640px) {{
    [class*="st-key-khung"] {{ padding: 12px; border-radius: 10px; }}
    .cau-to-mau {{ font-size: 1rem !important; line-height: 2 !important; }}
}}
</style>
""", unsafe_allow_html=True)


def khung(ten):
    """Một khung nổi (card) có nền khác nền chính. `ten` phải là duy nhất."""
    return st.container(key=f"khung_{ten}")


def trang_tri_bieu_do(fig):
    """Biểu đồ: vùng vẽ trắng, phần ngoài trong suốt để hòa vào khung."""
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#ffffff", font_color=CHU_CHINH,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ====================== GỌI BACKEND ======================
def lay_thong_bao_loi(r):
    """FastAPI trả 'detail' có thể là chuỗi hoặc danh sách lỗi validation."""
    try:
        detail = r.json().get("detail")
    except ValueError:
        return f"Lỗi {r.status_code} từ server."
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return f"Lỗi {r.status_code} từ server."


def goi_api(method, duong_dan, json=None, params=None):
    """Gọi Backend và trả về JSON. Nếu lỗi thì raise RuntimeError với thông báo tiếng Việt."""
    url = st.session_state["base_url"].rstrip("/") + duong_dan
    try:
        r = requests.request(method, url, json=json, params=params, timeout=60)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Không kết nối được tới Backend tại {url}. Kiểm tra server FastAPI đã chạy chưa.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Backend phản hồi quá lâu (quá 60 giây).")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Không gửi được yêu cầu đến Backend: {e}") from e
    if r.status_code == 422:
        raise RuntimeError("Dữ liệu không hợp lệ: " + lay_thong_bao_loi(r))
    if not 200 <= r.status_code < 300:
        raise RuntimeError(f"Server gặp lỗi ({r.status_code}): {lay_thong_bao_loi(r)}")
    try:
        return r.json()
    except ValueError:
        raise RuntimeError("Backend trả về dữ liệu không phải JSON.")


def chuan_hoa_nhan(nhan):
    """Chỉ nhận ba nhãn sentiment hợp lệ; nhãn lạ trả về None."""
    s = str(nhan or "").strip().lower()
    return {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}.get(s)


def nhan_tong_the(nhan):
    """Như chuan_hoa_nhan nhưng trả None khi Backend không dự đoán được (overall_sentiment = null)."""
    return None if nhan is None or not str(nhan).strip() else chuan_hoa_nhan(nhan)


def chu_nhan(nhan):
    """'Positive (Tích cực)' hoặc 'Không xác định'."""
    return f"{nhan} ({TEN_VN[nhan]})" if nhan in TEN_VN else KHONG_RO


def phan_tram(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "—"


def tham_so_loc(san_pham, tu_ngay, den_ngay):
    """Query params dùng chung cho GET /reviews và GET /statistics."""
    p = {"from_date": tu_ngay.isoformat(), "to_date": den_ngay.isoformat()}
    if san_pham != "Tất cả":
        p["product"] = san_pham
    return p


# ====================== HIỂN THỊ ======================
def to_mau_cau(noi_dung, ds_khia_canh):
    """
    Bọc từng cụm khía cạnh bằng <mark> đúng vị trí [start, end).
    Python cắt chuỗi theo ký tự Unicode, giống Backend, nên emoji không làm lệch vị trí.
    """
    hop_le = [a for a in ds_khia_canh
              if isinstance(a.get("start"), int) and isinstance(a.get("end"), int)
              and 0 <= a["start"] < a["end"] <= len(noi_dung)]
    hop_le.sort(key=lambda a: a["start"])

    phan, vi_tri = [], 0
    for a in hop_le:
        if a["start"] < vi_tri:          # bỏ qua cụm bị chồng lấn
            continue
        s = chuan_hoa_nhan(a.get("polarity"))
        mau_nen = MAU_NEN.get(s, "#e2e8f0")
        mau_chu = MAU_CHU.get(s, "#64748b")
        ten_nhan = TEN_VN.get(s, KHONG_RO)
        phan.append(html.escape(noi_dung[vi_tri:a["start"]]))
        tooltip = f'{a.get("category")} · {ten_nhan} · {phan_tram(a.get("confidence"))}'
        phan.append(
            f'<mark title="{html.escape(tooltip)}" style="background:{mau_nen};color:#111;'
            f'border-bottom:3px solid {mau_chu};padding:1px 4px;border-radius:4px;">'
            f'{html.escape(noi_dung[a["start"]:a["end"]])}'
            f'<sup style="color:{mau_chu};font-size:0.65em;margin-left:2px">'
            f'{html.escape(str(a.get("category")))}</sup></mark>'
        )
        vi_tri = a["end"]
    phan.append(html.escape(noi_dung[vi_tri:]))
    return (f'<div class="cau-to-mau" style="font-size:1.1rem;line-height:2.2;padding:12px 14px;'
            f'background:#ffffff;color:{CHU_CHINH};border:1px solid {VIEN_KHUNG};border-radius:8px;'
            f'word-wrap:break-word;overflow-wrap:anywhere;margin:8px 0">' + "".join(phan) + "</div>")


def chu_giai_mau():
    o = "".join(
        f'<span style="background:{MAU_NEN[k]};border-bottom:3px solid {MAU_CHU[k]};'
        f'color:#111;padding:2px 8px;border-radius:4px;margin-right:8px;display:inline-block;margin-bottom:4px">{v}</span>'
        for k, v in TEN_VN.items()
    )
    st.markdown("**Chú giải:** " + o, unsafe_allow_html=True)


def hien_thi_ket_qua(stt, noi_dung, kq, loi, da_luu):
    with khung(f"kq_{stt}"):
        if loi:
            st.error(f"❌ {loi}")
            st.caption(f"Review: {noi_dung[:150]}")
            return

        s = nhan_tong_the(kq.get("overall_sentiment"))
        if s:
            st.markdown(
                f'Cảm xúc tổng thể: <b style="color:{MAU_CHU[s]};font-size:1.3rem">{chu_nhan(s)}</b>'
                f' &nbsp; Độ tin cậy: <b>{phan_tram(kq.get("overall_confidence"))}</b>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'Cảm xúc tổng thể: <b style="color:#64748b;font-size:1.3rem">{KHONG_RO}</b>',
                        unsafe_allow_html=True)
            st.caption("Mô hình không tách được từ nào trong review này để dự đoán.")
        if da_luu:
            st.caption(f'✅ Đã lưu vào Dashboard · Sản phẩm: {kq.get("product")}')
        if kq.get("truncated"):
            st.warning("⚠️ Review quá dài nên mô hình đã cắt bớt. Chỉ phần đầu được phân tích.")

        # Dùng chính text Backend trả về để vị trí start/end khớp tuyệt đối
        cau_goc = kq.get("text") or noi_dung
        ds = kq.get("aspects") or []
        st.markdown(to_mau_cau(cau_goc, ds), unsafe_allow_html=True)

        if not ds:
            st.info("Không nhận diện được khía cạnh nào trong review này.")
        else:
            bang = pd.DataFrame([{
                "Đoạn văn bản": a.get("text"),
                "Category": a.get("category"),
                "Sentiment": chu_nhan(chuan_hoa_nhan(a.get("polarity"))),
                "Confidence": phan_tram(a.get("confidence")),
            } for a in ds])
            st.dataframe(bang, hide_index=True, width="stretch")


# ====================== THANH BÊN ======================
st.title("Phân tích cảm xúc và khía cạnh trong đánh giá sản phẩm")


st.session_state["base_url"] = st.sidebar.text_input("Địa chỉ Backend", BASE_URL_MAC_DINH)
if st.sidebar.button("Kiểm tra kết nối"):
    try:
        goi_api("GET", "/api/v1/health")
        st.sidebar.success("Backend đang chạy")
    except RuntimeError as e:
        st.sidebar.error(str(e))

tab_phan_tich, tab_dashboard = st.tabs(["Phân tích", "Dashboard"])


# ---------- TAB 1: PHÂN TÍCH ----------
def bam_nut(loai):
    """Chụp giá trị widget ngay lúc nhấn nút, TRƯỚC khi khóa chúng ở lần rerun sau.

    Không lấy review từ st.text_area(disabled=True): widget có thể bị cập nhật/reset
    khi chuyển từ enabled sang disabled, dẫn đến dùng nhầm giá trị cũ.
    """
    st.session_state["yeu_cau"] = {
        "loai": loai,
        "text": st.session_state.get("noi_dung", ""),
        "product": st.session_state.get("san_pham", ""),
        "multi": bool(st.session_state.get("nhieu_review", False)),
    }


with tab_phan_tich:
    # Có yêu cầu đang chờ -> khóa ô nhập và 2 nút trong suốt lúc gọi Backend (tránh lưu trùng)
    yeu_cau = st.session_state.pop("yeu_cau", None)
    dang_chay = yeu_cau is not None

    with khung("nhap"):
        san_pham = st.text_input("Tên sản phẩm (cần khi bấm Phân tích và Lưu)", key="san_pham")
        noi_dung = st.text_area("Nhập review tiếng Việt", height=140, key="noi_dung",
                                placeholder="Ví dụ: Pin dùng rất lâu nhưng camera chụp ban đêm hơi mờ.")
        nhieu_review = st.checkbox("Mỗi dòng là một review riêng (phân tích nhiều review một lần)",
                                   key="nhieu_review")

        b1, b2 = st.columns(2)
        b1.button("Phân tích", width="stretch", disabled=dang_chay,
                  on_click=bam_nut, args=("phan_tich",))
        b2.button("Phân tích và Lưu", type="primary", width="stretch", disabled=dang_chay,
                  on_click=bam_nut, args=("luu",))

    if dang_chay:
        # Luôn dùng snapshot lúc người dùng nhấn nút, không dùng widget bị disable.
        luu = yeu_cau["loai"] == "luu"
        noi_dung_gui = yeu_cau["text"]
        san_pham_gui = yeu_cau["product"]
        nhieu_review_gui = yeu_cau["multi"]
        if nhieu_review_gui:
            danh_sach = [d.strip() for d in noi_dung_gui.splitlines() if d.strip()]
        else:
            danh_sach = [noi_dung_gui.strip()] if noi_dung_gui.strip() else []

        # Xóa kết quả cũ ngay cả khi lần nhấn mới không hợp lệ.
        st.session_state["ket_qua"] = []
        st.session_state["ket_qua_dau_vao"] = None
        if not danh_sach:
            st.session_state["canh_bao"] = "Vui lòng nhập nội dung review trước khi phân tích."
        elif luu and not san_pham_gui.strip():
            st.session_state["canh_bao"] = "Vui lòng nhập tên sản phẩm để lưu review."
        else:
            ket_qua_moi = []
            with st.spinner(f"Đang phân tích {len(danh_sach)} review bằng PhoBERT..."):
                for t in danh_sach:
                    try:
                        if luu:
                            kq = goi_api("POST", "/api/v1/reviews",
                                         json={"product": san_pham_gui.strip(), "text": t})
                        else:
                            kq = goi_api("POST", "/api/v1/analyze", json={"text": t})
                        # Span offsets phụ thuộc vào từng ký tự của input. Tuyệt đối không
                        # hiển thị prediction cho một review khác khi backend trả sai text.
                        text_phan_hoi = kq.get("text")
                        if text_phan_hoi is not None and text_phan_hoi != t:
                            raise RuntimeError(
                                "Backend trả về kết quả của review khác với review vừa gửi. "
                                "Đã chặn hiển thị để tránh gắn nhầm nhãn; hãy kiểm tra API/backend."
                            )
                        ket_qua_moi.append((t, kq, None, luu))
                    except RuntimeError as e:
                        ket_qua_moi.append((t, None, str(e), False))
            st.session_state["ket_qua"] = ket_qua_moi
            st.session_state["ket_qua_dau_vao"] = (noi_dung_gui, nhieu_review_gui)
        st.rerun()          # vẽ lại để mở khóa nút; Dashboard cũng tải lại dữ liệu mới

    if st.session_state.get("canh_bao"):
        st.warning(st.session_state.pop("canh_bao"))

    ket_qua_hien_tai = st.session_state.get("ket_qua")
    dau_vao_da_gui = st.session_state.get("ket_qua_dau_vao")
    # Khi sửa ô nhập, ẩn kết quả cũ; không gây hiểu nhầm là kết quả của review mới.
    if ket_qua_hien_tai and dau_vao_da_gui != (noi_dung, nhieu_review):
        st.info("Bạn đã thay đổi nội dung review. Hãy bấm **Phân tích** hoặc "
                "**Phân tích và Lưu** để xem kết quả mới.")
    elif ket_qua_hien_tai:
        st.subheader("Kết quả")
        chu_giai_mau()
        for i, (t, kq, loi, da_luu) in enumerate(ket_qua_hien_tai):
            hien_thi_ket_qua(i, t, kq, loi, da_luu)

# ---------- TAB 2: DASHBOARD ----------
with tab_dashboard:
    try:
        # Lấy toàn bộ review (không lọc) chỉ để có danh sách tên sản phẩm cho ô chọn
        with st.spinner("Đang tải dữ liệu..."):
            tat_ca = goi_api("GET", "/api/v1/reviews")["reviews"]
    except RuntimeError as e:
        st.error(f"❌ {e}")
        st.stop()

    if not tat_ca:
        st.info("Chưa có review nào được lưu. Hãy sang tab **Phân tích** và bấm **Phân tích và Lưu**.")
        st.stop()

    # --- Bộ lọc ---
    # Backend lọc theo NGÀY của created_at (giờ UTC), nên mặc định và khoảng ngày cũng lấy theo UTC.
    ngay_utc = [pd.to_datetime(r["created_at"], utc=True).date() for r in tat_ca]
    with khung("loc"):
        f1, f2 = st.columns(2)
        ds_sp = ["Tất cả"] + sorted({r["product"] for r in tat_ca})
        chon_sp = f1.selectbox("Sản phẩm", ds_sp)
        khoang = f2.date_input("Khoảng thời gian (ngày lưu review)",
                value=(min(ngay_utc), max(max(ngay_utc), date.today())))
    if not khoang:
        st.info("Hãy chọn ít nhất một ngày để lọc.")
        st.stop()
    tu_ngay, den_ngay = (khoang[0], khoang[1]) if len(khoang) == 2 else (khoang[0], khoang[0])

    # --- Gọi 2 API với CÙNG bộ lọc ---
    params = tham_so_loc(chon_sp, tu_ngay, den_ngay)
    try:
        with st.spinner("Đang tải thống kê..."):
            thong_ke = goi_api("GET", "/api/v1/statistics", params=params)
            ds_review = goi_api("GET", "/api/v1/reviews", params=params)["reviews"]
    except RuntimeError as e:
        st.error(f"❌ {e}")
        st.stop()

    tong = thong_ke.get("total_reviews", 0)
    if tong == 0:
        st.info("Không có review nào khớp với bộ lọc đã chọn.")
        st.stop()

    # --- Số liệu tổng ---
    pb = thong_ke.get("overall_distribution", {})
    with khung("so_lieu"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng số review", tong)
        for cot, khoa, nhan in zip([m2, m3, m4], ["positive", "neutral", "negative"],
                                   ["Positive", "Neutral", "Negative"]):
            cot.metric(TEN_VN[nhan], f'{pb.get(khoa, 0)} ({pb.get(khoa, 0) / tong:.0%})')

    # --- Biểu đồ phân bố cảm xúc ---
    g1, g2 = st.columns(2)
    nhan = ["Positive", "Neutral", "Negative"]
    gia_tri = [pb.get(n.lower(), 0) for n in nhan]
    with g1, khung("bd_tron"):
        st.plotly_chart(trang_tri_bieu_do(px.pie(names=[TEN_VN[n] for n in nhan], values=gia_tri,
                                                 color=[TEN_VN[n] for n in nhan],
                                                 color_discrete_map={TEN_VN[n]: MAU_CHU[n] for n in nhan},
                                                 hole=0.4, title="Phân bố cảm xúc tổng thể")), width="stretch")

    # --- Biểu đồ khía cạnh ---
    kc = pd.DataFrame(thong_ke.get("aspect_statistics", []))
    for cot_dl in ("positive", "neutral", "negative"):       # phòng Backend thiếu cột
        if not kc.empty and cot_dl not in kc:
            kc[cot_dl] = 0
    with g2, khung("bd_khia_canh"):
        if kc.empty:
            st.info("Các review trong bộ lọc chưa có khía cạnh nào.")
        else:
            dai = kc.melt(id_vars="category", value_vars=["positive", "neutral", "negative"],
                          var_name="sentiment", value_name="so_lan")
            dai["sentiment"] = dai["sentiment"].str.capitalize().map(TEN_VN)
            fig_kc = px.bar(dai, x="so_lan", y="category", color="sentiment", orientation="h",
                            color_discrete_map={TEN_VN[n]: MAU_CHU[n] for n in nhan}, barmode="stack",
                            title="Số lần nhắc tới theo khía cạnh",
                            labels={"category": "", "so_lan": "Số lần", "sentiment": ""})
            trang_tri_bieu_do(fig_kc)
            fig_kc.update_layout(yaxis={"categoryorder": "total ascending"},       # nhiều nhất nằm trên cùng
                                 legend=dict(orientation="h", y=-0.2, x=0))        # chú thích xuống dưới
            st.plotly_chart(fig_kc, width="stretch")

    if not kc.empty:
        k1, k2 = st.columns(2)
        for cot, cot_dl, tieu_de, mau in [(k1, "positive", "Khía cạnh được khen nhiều nhất", MAU_CHU["Positive"]),
                                          (k2, "negative", "Khía cạnh bị chê nhiều nhất", MAU_CHU["Negative"])]:
            with cot, khung(f"top_{cot_dl}"):
                top = kc[kc[cot_dl] > 0].sort_values(cot_dl, ascending=False).head(5)
                if top.empty:
                    st.markdown(f"**{tieu_de}**")
                    st.info("Chưa có dữ liệu.")
                else:
                    fig = px.bar(top, x=cot_dl, y="category", orientation="h", title=tieu_de,
                                 color_discrete_sequence=[mau], labels={cot_dl: "Số lần", "category": "Khía cạnh"})
                    trang_tri_bieu_do(fig)
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, width="stretch")

    # --- Bảng danh sách review ---
    with khung("bang_review"):
        st.subheader("Danh sách review")
        bang = pd.DataFrame([{
            "Thời gian lưu (giờ VN)": pd.to_datetime(r["created_at"], utc=True)
                                        .tz_convert("Asia/Ho_Chi_Minh").strftime("%d/%m/%Y %H:%M"),
            "Sản phẩm": r["product"],
            "Nội dung": r["text"],
            "Cảm xúc": chu_nhan(nhan_tong_the(r.get("overall_sentiment"))),
            "Độ tin cậy": phan_tram(r.get("overall_confidence")),
        } for r in ds_review])
        st.dataframe(bang, hide_index=True, width="stretch")

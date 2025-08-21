import os, json, time, uuid, datetime, io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
import torch, sys
sys.path.append("/app")

from mnist_model import load_model, get_transforms, preprocess_to_28x28

SHARE = os.environ.get("SHARE_DIR", "/app/share_storage")
MODEL_PATH     = os.path.join(SHARE, "model", "model.pth")
LATEST_METRICS = os.path.join(SHARE, "metrics", "latest.json")
HISTORY        = os.path.join(SHARE, "metrics", "history.csv")
LOG_UPLOADS    = os.path.join(SHARE, "logs", "uploads.csv")
LOG_MISLABEL   = os.path.join(SHARE, "logs", "mislabels.csv")
BACKUP_DIR     = os.path.join(SHARE, "backupdata")
DATA_DIR       = os.path.join(SHARE, "data")

st.set_page_config(page_title="MNIST Feedback Loop", layout="centered")
st.title(":brain: MNIST Feedback Loop")

VERSION_JSON = os.path.join(SHARE, "model", "version.json")

def _read_model_version() -> int:
    try:
        if os.path.exists(VERSION_JSON):
            return int(json.load(open(VERSION_JSON)).get("version", 0))
    except:
        pass
    return 0

@st.cache_resource
def _load_model_cached(model_path: str, version_key: int):
    # version_key가 바뀌면 캐시 무효화 → 새 모델 로드
    return load_model(model_path)

def _append_csv(path, row_dict, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    need_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if need_header: f.write(header + "\n")
        f.write(",".join(str(row_dict.get(k, "")) for k in header.split(",")) + "\n")

def save_png(dst_dir, image: Image.Image, fname: str):
    os.makedirs(dst_dir, exist_ok=True)
    image.save(os.path.join(dst_dir, fname), format="PNG")

# ----- Sidebar: 메트릭 카드만 유지 -----
st.sidebar.header(":chart_with_upwards_trend: Model performance")

def _load_hist(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    if "ts" in df.columns:
        df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        return None
    for col in ["accuracy_clean","accuracy_aug","loss_clean","loss_aug"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df

hist = _load_hist(HISTORY)

latest = {"version": 0, "accuracy_clean": 0.0, "accuracy_aug": 0.0, "loss_clean": 0.0, "loss_aug": 0.0}
if os.path.exists(LATEST_METRICS):
    try: latest = json.load(open(LATEST_METRICS))
    except: pass

col1, col2 = st.sidebar.columns(2)
col1.metric("Clean Acc", f"{latest.get('accuracy_clean',0.0)*100:.2f}%")
col2.metric("Clean Loss", f"{latest.get('loss_clean',0.0):.4f}")
st.sidebar.caption(
    f"Model v{latest.get('version',0)} | "
    f"Robust Acc: {latest.get('accuracy_aug',0.0)*100:.2f}% | "
    f"Robust Loss: {latest.get('loss_aug',0.0):.4f}"
)

# ----- Main: 업로드/피드백 -----
name = st.text_input("Your name", value="")
upload = st.file_uploader("Upload a digit photo (0–9)", type=["png","jpg","jpeg"])

@st.cache_resource
def _tfm(): return get_transforms()

if upload and name.strip():
    raw = upload.read()

    # 화면에는 원본 이미지 표시
    pil_orig = Image.open(io.BytesIO(raw))
    st.image(pil_orig, caption="Uploaded image (original)", width=200)

    # 모델 입력은 전처리된 28x28 사용
    pil28 = preprocess_to_28x28(raw)

    model_version = _read_model_version()
    model = _load_model_cached(MODEL_PATH, model_version)
    x = _tfm()(pil28).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy().ravel()
    pred, conf = int(probs.argmax()), float(probs.max())

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted", f"{pred}")
    c2.metric("Confidence", f"{conf*100:.2f}%")

    ts_now = int(time.time())
    dt_now = datetime.datetime.fromtimestamp(ts_now).isoformat()
    uid = uuid.uuid4().hex
    
    # 세션 상태에 저장 (원본+전처리 둘 다 보관)
    st.session_state["upload_info"] = {
        "raw": raw, "pil_orig": pil_orig, "pil28": pil28,
        "name": name, "pred": pred, "conf": conf, "ts": ts_now, "dt": dt_now, "uid": uid
    }
    
    col_ok, col_ng = st.columns(2)
    btn_ok = col_ok.button("맞았어 :white_check_mark:")
    btn_ng = col_ng.button("틀렸어 :x:")

    if btn_ok:
        info = st.session_state["upload_info"]
        true_label = info["pred"]
        fname = f"{info['ts']}_{info['name']}_{info['uid']}.png".replace(",", "_").replace(" ", "_")
        
        # 맞았을 때: backupdata에는 '원본' 저장
        save_png(os.path.join(BACKUP_DIR, str(true_label)), info["pil_orig"], fname)
        _append_csv(LOG_UPLOADS,
            {"ts": info['ts'], "datetime": info['dt'], "name": info['name'], "filename": fname,
             "pred": info['pred'], "confidence": round(info['conf'], 4),
             "is_correct": 1, "true_label": true_label},
            "ts,datetime,name,filename,pred,confidence,is_correct,true_label")
        st.success(f"Saved to backupdata/{true_label}/")
        del st.session_state["upload_info"]
        st.experimental_rerun()

    if btn_ng:
        st.session_state["wrong_label_form_active"] = True
    
    if "wrong_label_form_active" in st.session_state and st.session_state["wrong_label_form_active"]:
        info = st.session_state["upload_info"]
        with st.form("confirm_wrong_label"):
            true_label = st.number_input("정답(0~9)을 입력해 주세요", min_value=0, max_value=9, value=info["pred"], step=1)
            submitted = st.form_submit_button("정답 확정 및 저장")
            if submitted:
                fname = f"{info['ts']}_{info['name']}_{info['uid']}.png".replace(",", "_").replace(" ", "_")

                # 틀렸을 때: backupdata에는 '원본', data에는 '전처리 28x28' 저장
                save_png(os.path.join(BACKUP_DIR, str(true_label)), info["pil_orig"], fname)
                save_png(os.path.join(DATA_DIR, str(true_label)), info["pil28"], fname)

                _append_csv(LOG_MISLABEL,
                    {"ts": info['ts'], "datetime": info['dt'], "name": info['name'],
                     "filename": fname, "pred": info['pred'], "confidence": round(info['conf'], 4),
                     "true_label": true_label},
                    "ts,datetime,name,filename,pred,confidence,true_label")
                _append_csv(LOG_UPLOADS,
                    {"ts": info['ts'], "datetime": info['dt'], "name": info['name'],
                     "filename": fname, "pred": info['pred'], "confidence": round(info['conf'], 4),
                     "is_correct": 0, "true_label": true_label},
                    "ts,datetime,name,filename,pred,confidence,is_correct,true_label")
                st.success(f"Saved to backupdata/{true_label}/ and data/{true_label}/")
                del st.session_state["wrong_label_form_active"]
                del st.session_state["upload_info"]
                st.experimental_rerun()

# ----- Main 하단: 성능 히스토리 그래프 -----
st.markdown("---")
st.subheader(":bar_chart: Model performance history")

if hist is not None and len(hist):
    # 날짜 정규화 + 같은 날 여러 행이면 '마지막' 것만 남기기
    if "ts" in hist.columns:
        hist = hist.sort_values("ts")
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    hist_day = hist.drop_duplicates(subset=["date"], keep="last")

    # 오늘 기준 ±3일(총 7일) 구간
    today = pd.Timestamp(datetime.date.today())
    start = today - pd.Timedelta(days=3)
    end   = today + pd.Timedelta(days=3)

    # 7일 창에 해당하는 데이터만 필터 (없으면 최근 7개)
    win = hist_day[(hist_day["date"] >= start) & (hist_day["date"] <= end)]
    if win.empty:
        win = hist_day.tail(7)

    fig, ax1 = plt.subplots(figsize=(6,3))
    ax1.plot(win["date"], win["loss_clean"], color="red", marker="x", label="Loss (Clean)")
    ax1.set_ylabel("Loss", color="red"); ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xlabel("Date"); ax1.grid(True)

    # x축을 정확히 7일 창으로 고정 (미래 날짜도 눈금 표시) + MM-DD 형식
    ax1.set_xlim([start, end])
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax2 = ax1.twinx()
    ax2.plot(win["date"], win["accuracy_clean"]*100.0, color="green", marker="o", label="Accuracy (Clean)")
    ax2.set_ylabel("Accuracy (%)", color="green"); ax2.tick_params(axis="y", labelcolor="green")

    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("No history yet.")

st.caption("""
- 재학습은 매일 11:50(KST)에 1회 수행됩니다.
- 위 꺾은선: x=날짜, y=Accuracy(초록)/Loss(빨강) — Clean 기준.
- Robust 지표(증강 테스트: 회전±15° + 노이즈 0.15)는 사이드바 카드에 표기됩니다.
- 업로드는 backupdata/<label>/, 오답은 data/<label>/에도 저장되어 재학습에 사용됩니다.
""")

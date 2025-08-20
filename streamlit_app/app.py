import os, json, time, uuid, datetime, io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torch, sys
import matplotlib.dates as mdates
import matplotlib.ticker as mticker




# app.py 기준으로 modeling 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
modeling_path = os.path.join(current_dir, "..", "modeling")
sys.path.append(modeling_path)

from mnist_model import load_model, get_transforms, preprocess_to_28x28


# ---------------- Directories ----------------
SHARE = os.environ.get("SHARE_DIR", "/app/share_storage")
MODEL_PATH     = os.path.join(SHARE, "model", "model.pth")
LATEST_METRICS = os.path.join(SHARE, "metrics", "latest.json")
# HISTORY        = os.path.join(SHARE, "metrics", "history.csv")
HISTORY = "/Users/lyra8/final_docker_mlops/share_storage/metrics/history.csv"
LOG_UPLOADS    = os.path.join(SHARE, "logs", "uploads.csv")
LOG_MISLABEL   = os.path.join(SHARE, "logs", "mislabels.csv")
BACKUP_DIR     = os.path.join(SHARE, "backupdata")
DATA_DIR       = os.path.join(SHARE, "data")
VERSION_JSON   = os.path.join(SHARE, "model", "version.json")

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="MNIST Feedback Loop", layout="centered")
st.title("😎 MNIST Feedback Loop")  # 원하시면 이모티콘 바꾸면 됩니다

# ---------------- Utils ----------------
def _read_model_version() -> int:
    try:
        if os.path.exists(VERSION_JSON):
            return int(json.load(open(VERSION_JSON)).get("version", 0))
    except:
        pass
    return 0

@st.cache_resource
def _load_model_cached(model_path: str, version_key: int):
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

@st.cache_resource
def _tfm(): return get_transforms()


# ---------------- Sidebar: metrics ----------------
st.sidebar.header("📈 Model performance")

def _load_hist(path):
    # CSV가 없으면 빈 DataFrame 반환
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ts","date","time","version",
                                     "accuracy_clean","accuracy_aug",
                                     "loss_clean","loss_aug"])
    
    df = pd.read_csv(path)
    
    # 날짜 컬럼 처리
    if "ts" in df.columns:
        df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        df["date"] = pd.NA  # 없으면 NaT로 처리
    
    # 필요한 컬럼 체크, 없으면 추가
    for col in ["accuracy_clean","accuracy_aug","loss_clean","loss_aug"]:
        if col not in df.columns:
            df[col] = pd.NA
    
    return df


# def _load_hist(path):
#     if not os.path.exists(path): return None
#     df = pd.read_csv(path)
#     if "ts" in df.columns:
#         df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date
#     elif "date" in df.columns:
#         df["date"] = pd.to_datetime(df["date"]).dt.date
#     else:
#         return None
#     for col in ["accuracy_clean","accuracy_aug","loss_clean","loss_aug"]:
#         if col not in df.columns:
#             df[col] = pd.NA
#     return df

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

# ---------------- Main: Upload / Feedback ----------------
name = st.text_input("Your name", value="")
upload = st.file_uploader("Upload a digit photo (0–9)", type=["png","jpg","jpeg"])

if upload and name.strip():
    raw = upload.read()
    
    # 1) 화면에 보여주는 원본
    pil_orig = Image.open(io.BytesIO(raw))
    st.image(pil_orig, caption="Uploaded image", width=200)
    
    # 2) 모델 입력용: 흑백 + 28x28 전처리 후 Tensor 변환
    pil_model = pil_orig.convert("L")  # 흑백
    pil_model = preprocess_to_28x28(pil_model)  # PIL Image 반환
    tfm = _tfm()  # transform 함수
    x = tfm(pil_model).unsqueeze(0)  # Tensor로 변환 + 배치 차원 추가
    
    model_version = _read_model_version()
    model = _load_model_cached(MODEL_PATH, model_version)
    
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
    
    st.session_state["upload_info"] = {
        "raw": raw, "pil_orig": pil_orig, "name": name,
        "pred": pred, "conf": conf, "ts": ts_now, "dt": dt_now, "uid": uid
    }


    
    col_ok, col_ng = st.columns(2)
    btn_ok = col_ok.button("맞았어 ✅")
    btn_ng = col_ng.button("틀렸어 ❌")
    
    if btn_ok:
        info = st.session_state["upload_info"]
        true_label = info["pred"]
        fname = f"{info['ts']}_{info['name']}_{info['uid']}.png".replace(",", "_").replace(" ", "_")
        
        save_png(os.path.join(BACKUP_DIR, str(true_label)), info["pil_orig"], fname)
        _append_csv(LOG_UPLOADS,
            {"ts": info['ts'], "datetime": info['dt'], "name": info['name'], "filename": fname,
             "pred": info['pred'], "confidence": round(info['conf'], 4), "is_correct": 1, "true_label": true_label},
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
                
                # backupdata에는 원본, data에는 전처리된 28x28 저장
                save_png(os.path.join(BACKUP_DIR, str(true_label)), info["pil_orig"], fname)
                pil28 = preprocess_to_28x28(info["pil_orig"].convert("L"))
                save_png(os.path.join(DATA_DIR, str(true_label)), pil28, fname)
                
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

# ---------------- Main 하단: Performance History ----------------


st.markdown("---")
st.subheader("📊 Model performance history")
# 날짜 타입 변환
hist["date"] = pd.to_datetime(hist["date"])


fig, ax1 = plt.subplots(figsize=(6,3))
ax1.plot(hist["date"], hist["loss_clean"], color="red", marker="x", label="Loss (Clean)")
ax1.set_ylabel("Loss", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.set_xlabel("Date")
ax1.grid(True)

# x축: 오늘 기준 최근 5일
today = pd.to_datetime(datetime.date.today())
last_5days = [today - pd.Timedelta(days=i) for i in reversed(range(5))]
ax1.set_xlim([last_5days[0], last_5days[-1]])

# x축 라벨 고정
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.autofmt_xdate()

ax2 = ax1.twinx()
ax2.plot(hist["date"], hist["accuracy_clean"]*100.0, color="green", marker="o", label="Accuracy (Clean)")
ax2.set_ylabel("Accuracy (%)", color="green")
ax2.tick_params(axis="y", labelcolor="green")

fig.tight_layout()
st.pyplot(fig)

st.caption("""
- 재학습은 매일 15:00(KST)에 1회 수행됩니다.
- 위 꺾은선: x=날짜, y=Accuracy(초록)/Loss(빨강) — Clean 기준.
- Robust 지표(증강 테스트: 회전±15° + 노이즈 0.15)는 사이드바 카드에 표기됩니다.
- 업로드는 backupdata/<label>/, 오답은 data/<label>/에도 저장되어 재학습에 사용됩니다.
""")

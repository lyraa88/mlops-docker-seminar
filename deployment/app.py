import streamlit as st
from PIL import Image
import os
import datetime
from modeling.model import predict  # (예측값, 정확도) 반환한다고 가정

st.set_page_config(page_title="사진 업로드 모델 데모", page_icon="📷", layout="centered")

# CSS: 버튼 꾸미기
st.markdown(
    """
    <style>
    div[data-testid="column"]:nth-of-type(1) button {
        background-color: #2196F3 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
    div[data-testid="column"]:nth-of-type(1) button:hover {
        background-color: #1976D2 !important;
        color: white !important;
    }
    div[data-testid="column"]:nth-of-type(2) button {
        background-color: #F44336 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
    div[data-testid="column"]:nth-of-type(2) button:hover {
        background-color: #D32F2F !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 📷 사진 업로드 모델 데모")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    if st.button("🔍 내 손글씨 예측해보기"):
        # 모델 예측 (결과, 정확도 % 반환)
        result, accuracy = predict(image)
        st.session_state["model_result"] = result
        st.session_state["model_accuracy"] = accuracy
        st.session_state["uploaded_image"] = uploaded_file

# 예측 결과 & 정확도 표시
if "model_result" in st.session_state:
    st.markdown(
        f"""
        <div style='padding:10px; background:#f0f0f0; border-radius:8px; 
                    border:1px solid #ddd; text-align:center; font-weight:bold;'>
        모델 예측 결과: {st.session_state['model_result']}<br>
        <span style='color:#555;'>({st.session_state['model_accuracy']}% 정확도를 기반으로 예측)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 정답이야!"):
            st.success("감사합니다! 👍")

    with col2:
        if st.button("❌ 틀렸어!"):
            st.session_state["show_correct_input"] = True

# 정답 입력 시 이미지 저장 (정답 텍스트 저장 제거)
if st.session_state.get("show_correct_input", False):
    correct_answer = st.text_input("정답을 입력해주세요:")
    if correct_answer:
        folder_path = os.path.join("data", correct_answer)
        os.makedirs(folder_path, exist_ok=True)

        if "uploaded_image" in st.session_state:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(st.session_state["uploaded_image"].name)[1]
            save_path = os.path.join(folder_path, f"{now}{ext}")

            image = Image.open(st.session_state["uploaded_image"])
            image.save(save_path)
            st.success(f"이미지를 '{folder_path}' 폴더에 저장했습니다")

        # 정답 텍스트 파일 저장 제거
        st.session_state["show_correct_input"] = False
        st.session_state.pop("uploaded_image", None)

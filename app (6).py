import streamlit as st
import numpy as np, cv2, tempfile, os, pandas as pd
from PIL import Image
import tensorflow as tf
import mediapipe as mp

st.set_page_config(page_title='Elderly Fall Detection', page_icon='🧓', layout='wide')
st.title('🧓 AI-Powered Elderly Fall Detection')
st.write('Upload an image or a video. The MobileNetV2 CNN predicts the activity and '
         'MediaPipe Pose draws the skeleton. A fall raises an alert.')

CLASSES = ['Fall', 'Normal', 'Sitting', 'Standing', 'Walking']
MODEL_PATH = 'fall_mobilenetv2_final.h5'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

@st.cache_resource
def load_pose():
    return mp.solutions.pose.Pose(static_image_mode=True)

model = load_model()
pose = load_pose()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def predict_frame(img_bgr):
    img = cv2.resize(img_bgr, (224, 224))
    x = img.astype('float32') / 255.0
    if model is not None:
        prob = model.predict(x[None, ...], verbose=0)[0]
        idx = int(np.argmax(prob))
        return CLASSES[idx], prob
    # fallback geometry heuristic using MediaPipe
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        vert = abs(lm[23].y - lm[25].y)
        return ('Fall' if vert < 0.18 else 'Standing', None)
    return ('Normal', None)

def overlay_pose(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

st.sidebar.header('Input')
src = st.sidebar.radio('Choose source', ['Image', 'Video'])

# monitoring log (session state)
if 'log' not in st.session_state:
    st.session_state.log = []

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader('Prediction')
    pred_box = st.empty()
    img_box = st.empty()
with col2:
    st.subheader('Monitoring')
    mon_box = st.empty()

if src == 'Image':
    up = st.sidebar.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
    if up is not None:
        img = Image.open(up).convert('RGB')
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        label, prob = predict_frame(bgr)
        shown = overlay_pose(bgr)
        img_box.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), channels='RGB')
        if label == 'Fall':
            pred_box.error('🚨 FALL DETECTED — ' + label)
        else:
            pred_box.success('Activity: ' + label)
        if prob is not None:
            st.sidebar.bar_chart({CLASSES[i]: float(prob[i]) for i in range(len(CLASSES))})
        st.session_state.log.append({'frame': len(st.session_state.log), 'label': label,
                                     'fall': label == 'Fall'})
else:
    up = st.sidebar.file_uploader('Upload video', type=['mp4', 'avi', 'mov'])
    if up is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(up.read()); tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        fall_count = 0; total = 0
        frame_slot = img_box; status = pred_box
        while True:
            ok, frame = cap.read()
            if not ok: break
            total += 1
            label, prob = predict_frame(frame)
            shown = overlay_pose(frame)
            frame_slot.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), channels='RGB')
            status.error('🚨 FALL DETECTED') if label == 'Fall' else status.info('Activity: ' + label)
            if label == 'Fall': fall_count += 1
            st.session_state.log.append({'frame': total, 'label': label, 'fall': label == 'Fall'})
        cap.release(); os.unlink(tfile.name)
        st.write(f'Processed {total} frames. Falls detected: {fall_count}')

# monitoring analytics
if st.session_state.log:
    ldf = pd.DataFrame(st.session_state.log)
    vc = ldf['label'].value_counts()
    mon_box.bar_chart(vc)
    falls = ldf[ldf['fall']]
    if len(falls):
        mon_box.write('Fall frames: ' + ', '.join(str(int(f)) for f in falls['frame'].tolist()))
else:
    mon_box.write('No predictions yet.')

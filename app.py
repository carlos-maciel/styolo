import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# Configuracao da pagina - mobile friendly
st.set_page_config(
    page_title="Styolo",
    page_icon="S",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS customizado para interface mobile-like
st.markdown("""
<style>
    /* Esconder menu e rodape do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Container principal */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 500px;
    }

    /* Estilo do titulo */
    .app-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }

    .app-subtitle {
        text-align: center;
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    /* Container da camera */
    .camera-container {
        background: #000;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    /* Estilo dos resultados */
    .result-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }

    .detection-count {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #28a745;
    }

    .detection-label {
        text-align: center;
        font-size: 0.85rem;
        color: #666;
    }

    /* Tabs customizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f0f0f0;
        border-radius: 10px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        flex: 1;
        border-radius: 8px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: white;
    }

    /* Camera input */
    .stCameraInput > div {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Slider */
    .stSlider {
        padding-top: 0.5rem;
    }

    /* Objeto detectado */
    .object-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Classes COCO para YOLO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Traducao das classes para portugues
CLASSES_PT = {
    'person': 'Pessoa', 'bicycle': 'Bicicleta', 'car': 'Carro', 'motorcycle': 'Moto',
    'airplane': 'Aviao', 'bus': 'Onibus', 'train': 'Trem', 'truck': 'Caminhao',
    'boat': 'Barco', 'traffic light': 'Semaforo', 'fire hydrant': 'Hidrante',
    'stop sign': 'Placa Pare', 'parking meter': 'Parquimetro', 'bench': 'Banco',
    'bird': 'Passaro', 'cat': 'Gato', 'dog': 'Cachorro', 'horse': 'Cavalo',
    'sheep': 'Ovelha', 'cow': 'Vaca', 'elephant': 'Elefante', 'bear': 'Urso',
    'zebra': 'Zebra', 'giraffe': 'Girafa', 'backpack': 'Mochila', 'umbrella': 'Guarda-chuva',
    'handbag': 'Bolsa', 'tie': 'Gravata', 'suitcase': 'Mala', 'frisbee': 'Frisbee',
    'skis': 'Esquis', 'snowboard': 'Snowboard', 'sports ball': 'Bola',
    'kite': 'Pipa', 'baseball bat': 'Taco', 'baseball glove': 'Luva',
    'skateboard': 'Skate', 'surfboard': 'Prancha', 'tennis racket': 'Raquete',
    'bottle': 'Garrafa', 'wine glass': 'Taca', 'cup': 'Copo', 'fork': 'Garfo',
    'knife': 'Faca', 'spoon': 'Colher', 'bowl': 'Tigela', 'banana': 'Banana',
    'apple': 'Maca', 'sandwich': 'Sanduiche', 'orange': 'Laranja', 'broccoli': 'Brocolis',
    'carrot': 'Cenoura', 'hot dog': 'Cachorro-quente', 'pizza': 'Pizza', 'donut': 'Rosquinha',
    'cake': 'Bolo', 'chair': 'Cadeira', 'couch': 'Sofa', 'potted plant': 'Planta',
    'bed': 'Cama', 'dining table': 'Mesa', 'toilet': 'Vaso', 'tv': 'TV',
    'laptop': 'Notebook', 'mouse': 'Mouse', 'remote': 'Controle', 'keyboard': 'Teclado',
    'cell phone': 'Celular', 'microwave': 'Microondas', 'oven': 'Forno', 'toaster': 'Torradeira',
    'sink': 'Pia', 'refrigerator': 'Geladeira', 'book': 'Livro', 'clock': 'Relogio',
    'vase': 'Vaso', 'scissors': 'Tesoura', 'teddy bear': 'Ursinho', 'hair drier': 'Secador',
    'toothbrush': 'Escova de dente'
}

def download_yolo_files():
    """Baixa os arquivos do modelo YOLO se nao existirem"""
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"

    weights_path = "yolov4.weights"
    config_path = "yolov4.cfg"

    if not os.path.exists(weights_path):
        st.info("Baixando modelo... Isso pode demorar alguns minutos.")
        try:
            urllib.request.urlretrieve(weights_url, weights_path)
        except Exception as e:
            st.error(f"Erro ao baixar modelo: {e}")
            return False

    if not os.path.exists(config_path):
        try:
            urllib.request.urlretrieve(config_url, config_path)
        except Exception as e:
            st.error(f"Erro ao baixar configuracao: {e}")
            return False

    return True

@st.cache_resource
def load_yolo_model():
    """Carrega o modelo YOLO usando cache do Streamlit"""
    weights_path = "yolov4.weights"
    config_path = "yolov4.cfg"

    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        if not download_yolo_files():
            return None

    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        return net
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

def detect_objects(image, net, confidence_threshold=0.5):
    """Detecta objetos na imagem usando YOLO"""
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    detected_objects = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label_en = COCO_CLASSES[class_ids[i]]
            label_pt = CLASSES_PT.get(label_en, label_en)
            conf = confidences[i]
            detected_objects.append({"label": label_pt, "label_en": label_en, "confidence": conf})

            # Desenhar retangulo com cor verde
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)

            # Fundo para o texto
            text = f"{label_pt}: {conf:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - text_h - 10), (x + text_w + 4, y), (0, 200, 0), -1)
            cv2.putText(image, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image, detected_objects

def main():
    # Titulo do app
    st.markdown('<p class="app-title">Styolo</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Deteccao de objetos com inteligencia artificial</p>', unsafe_allow_html=True)

    # Carregar modelo
    net = load_yolo_model()

    if net is None:
        st.error("Nao foi possivel carregar o modelo. Verifique sua conexao.")
        st.stop()

    # Configuracao de confianca (expansivel)
    with st.expander("Configuracoes"):
        confidence_threshold = st.slider(
            "Sensibilidade da deteccao",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Valores menores detectam mais objetos"
        )
        st.caption("Valores baixos: mais deteccoes | Valores altos: mais precisao")

    # Tabs para Camera e Upload
    tab_camera, tab_upload = st.tabs(["Camera", "Galeria"])

    img = None

    with tab_camera:
        camera_photo = st.camera_input(
            "Aponte para um objeto e tire a foto",
            label_visibility="collapsed"
        )

        if camera_photo is not None:
            file_bytes = camera_photo.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Selecione uma imagem",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Processar imagem
    if img is not None:
        with st.spinner("Analisando imagem..."):
            img_with_detections, detected_objects = detect_objects(img.copy(), net, confidence_threshold)
            img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)

        # Exibir resultado
        st.image(img_rgb, use_container_width=True)

        # Mostrar contagem
        if detected_objects:
            # Contador principal
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f'<p class="detection-count">{len(detected_objects)}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="detection-label">objeto{"s" if len(detected_objects) > 1 else ""} detectado{"s" if len(detected_objects) > 1 else ""}</p>', unsafe_allow_html=True)

            # Lista de objetos
            st.markdown("---")
            st.markdown("**Objetos encontrados:**")

            # Agrupar objetos iguais
            from collections import Counter
            labels = [obj["label"] for obj in detected_objects]
            label_counts = Counter(labels)

            # Mostrar como badges
            badges_html = ""
            for label, count in label_counts.items():
                if count > 1:
                    badges_html += f'<span class="object-badge">{label} ({count}x)</span>'
                else:
                    badges_html += f'<span class="object-badge">{label}</span>'

            st.markdown(badges_html, unsafe_allow_html=True)

            # Detalhes expansiveis
            with st.expander("Ver detalhes"):
                for i, obj in enumerate(detected_objects, 1):
                    st.markdown(f"{i}. **{obj['label']}** - {obj['confidence']:.0%} de confianca")

        else:
            st.warning("Nenhum objeto detectado. Tente aproximar a camera ou ajustar a sensibilidade.")

    else:
        # Estado inicial
        st.markdown("---")
        st.markdown(
            """
            **Como usar:**

            1. Tire uma foto usando a camera acima
            2. Ou selecione uma imagem da galeria
            3. Aguarde a analise automatica

            O sistema detecta mais de 80 tipos de objetos como pessoas, animais, veiculos, moveis e alimentos.
            """
        )

if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# Configuracao da pagina
st.set_page_config(
    page_title="Deteccao de Objetos - Styolo",
    page_icon="📷",
    layout="wide"
)

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

def download_yolo_files():
    """Baixa os arquivos do modelo YOLO se nao existirem"""
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"

    weights_path = "yolov4.weights"
    config_path = "yolov4.cfg"

    if not os.path.exists(weights_path):
        st.info("Baixando arquivo de pesos do YOLOv4... Isso pode demorar alguns minutos.")
        try:
            urllib.request.urlretrieve(weights_url, weights_path)
            st.success("Arquivo de pesos baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar arquivo de pesos: {e}")
            return False

    if not os.path.exists(config_path):
        st.info("Baixando arquivo de configuracao do YOLOv4...")
        try:
            urllib.request.urlretrieve(config_url, config_path)
            st.success("Arquivo de configuracao baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar arquivo de configuracao: {e}")
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
        st.error(f"Erro ao carregar modelo YOLO: {e}")
        return None

def detect_objects(image, net, confidence_threshold=0.5):
    """Detecta objetos na imagem usando YOLO"""
    height, width = image.shape[:2]

    # Criar blob da imagem
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obter deteccoes
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Processar deteccoes
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

    # Aplicar Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Lista de objetos detectados
    detected_objects = []

    # Desenhar bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = COCO_CLASSES[class_ids[i]]
            conf = confidences[i]
            detected_objects.append({"label": label, "confidence": conf})

            # Desenhar retangulo
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Desenhar label
            text = f"{label}: {conf:.2f}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, detected_objects

def main():
    st.title("📷 Deteccao de Objetos - Styolo")
    st.markdown("Tire uma foto ou faca upload de uma imagem para detectar objetos!")
    st.markdown("---")

    # Sidebar com configuracoes
    st.sidebar.title("⚙️ Configuracoes")

    # Verificar se o modelo esta disponivel
    with st.spinner("Carregando modelo YOLO..."):
        net = load_yolo_model()

    if net is None:
        st.error("Nao foi possivel carregar o modelo YOLO. Verifique se os arquivos estao disponiveis.")
        st.stop()

    st.sidebar.success("✅ Modelo YOLO carregado!")

    # Configuracoes do modelo
    confidence_threshold = st.sidebar.slider(
        "Limiar de Confianca",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Ajuste a sensibilidade da deteccao"
    )

    # Modo de operacao
    mode = st.sidebar.radio(
        "Fonte da Imagem",
        ["📷 Tirar Foto", "📁 Upload de Imagem"],
        help="Escolha como voce quer fornecer a imagem"
    )

    # Informacoes sobre o projeto
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Sobre o Projeto")
    st.sidebar.markdown("""
    Esta aplicacao utiliza:
    - **Streamlit** para interface web
    - **YOLOv4** para deteccao de objetos
    - **OpenCV** para processamento de imagem

    Detecta **80 classes** de objetos incluindo pessoas, animais, veiculos, objetos domesticos e mais!
    """)

    # Conteudo principal
    img = None

    if mode == "📷 Tirar Foto":
        st.markdown("### 📷 Tire uma Foto")
        st.markdown("Clique no botao abaixo para ativar a camera e tirar uma foto.")

        # Usar camera_input do Streamlit
        camera_photo = st.camera_input("Tire uma foto para detectar objetos")

        if camera_photo is not None:
            # Converter para OpenCV
            file_bytes = camera_photo.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    else:  # Upload de Imagem
        st.markdown("### 📁 Upload de Imagem")

        uploaded_file = st.file_uploader(
            "Escolha uma imagem",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Faca upload de uma imagem para detectar objetos"
        )

        if uploaded_file is not None:
            # Converter para OpenCV
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Processar imagem se disponivel
    if img is not None:
        st.markdown("---")

        with st.spinner("Detectando objetos..."):
            # Detectar objetos
            img_with_detections, detected_objects = detect_objects(img.copy(), net, confidence_threshold)

            # Converter para RGB para exibicao
            img_rgb_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb_detected = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)

        # Exibir resultados
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Imagem Original**")
            st.image(img_rgb_original, use_container_width=True)

        with col2:
            st.markdown("**Objetos Detectados**")
            st.image(img_rgb_detected, use_container_width=True)

        # Mostrar lista de objetos detectados
        st.markdown("---")

        if detected_objects:
            st.success(f"✅ {len(detected_objects)} objeto(s) detectado(s)!")

            # Criar tabela de resultados
            st.markdown("### 📊 Objetos Encontrados")

            cols = st.columns(min(len(detected_objects), 4))
            for i, obj in enumerate(detected_objects):
                with cols[i % 4]:
                    st.metric(
                        label=obj["label"].capitalize(),
                        value=f"{obj['confidence']*100:.1f}%"
                    )
        else:
            st.warning("Nenhum objeto detectado. Tente ajustar o limiar de confianca ou tire outra foto.")

        # Botao para nova deteccao
        st.markdown("---")
        st.info("💡 Para detectar novos objetos, tire outra foto ou faca upload de outra imagem.")

if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from PIL import Image
import os
import urllib.request

# Configuração da página
st.set_page_config(
    page_title="Detecção de Objetos em Tempo Real",
    page_icon="🎯",
    layout="wide"
)

# Configuração do WebRTC simplificada para melhor compatibilidade
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]}
    ],
    "iceCandidatePoolSize": 10,
    "iceTransportPolicy": "all"
})

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
    """Baixa os arquivos do modelo YOLO se não existirem"""
    # Usar YOLOv4 que é mais compatível com OpenCV
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
        st.info("Baixando arquivo de configuração do YOLOv4...")
        try:
            urllib.request.urlretrieve(config_url, config_path)
            st.success("Arquivo de configuração baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar arquivo de configuração: {e}")
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
    
    # Obter detecções
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    
    # Processar detecções
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
    
    # Desenhar bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{COCO_CLASSES[class_ids[i]]}: {confidences[i]:.2f}"
            
            # Desenhar retângulo
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Desenhar label
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.net = load_yolo_model()
        self.confidence_threshold = 0.5
        self.frame_count = 0
        self.skip_frames = 1  # Processar todos os frames para tempo real verdadeiro
    
    def recv(self, frame):
        if self.net is None:
            return frame
        
        # Para tempo real verdadeiro, processar todos os frames
        self.frame_count += 1
        
        try:
            # Converter frame para array numpy
            img = frame.to_ndarray(format="bgr24")
            
            # Detectar objetos em tempo real
            img = detect_objects(img, self.net, self.confidence_threshold)
            
            # Converter de volta para VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            # Em caso de erro, retornar frame original
            return frame

def main():
    st.title("Detecção de Objetos em Tempo Real")
    st.markdown("---")
    
    # Sidebar com configurações
    st.sidebar.title("Configurações")
    
    # Verificar se o modelo está disponível
    net = load_yolo_model()
    if net is None:
        st.error("Nao foi possivel carregar o modelo YOLO. Verifique se os arquivos estao disponiveis.")
        st.stop()
    
    st.sidebar.success("Modelo YOLO carregado com sucesso!")
    
    # Configurações do modelo
    confidence_threshold = st.sidebar.slider(
        "Limiar de Confianca",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Ajuste a sensibilidade da deteccao"
    )
    
    # Modo de operação - padrão para tempo real
    mode = st.sidebar.selectbox(
        "Modo de Operacao",
        ["WebRTC (Tempo Real)", "Upload de Imagem"],
        index=0,  # Sempre começar com tempo real
        help="WebRTC = Camera em tempo real | Upload = Imagem estatica"
    )
    
    # Informações sobre o projeto
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Sobre o Projeto")
    st.sidebar.markdown("""
    Esta aplicação utiliza:
    - **Streamlit** para interface web
    - **WebRTC** para captura de vídeo
    - **YOLOv3** para detecção de objetos
    - **OpenCV** para processamento de imagem
    """)
    
    # Instruções de uso
    st.markdown("### Como usar:")
    st.markdown("""
    **🎥 Modo WebRTC (Tempo Real) - RECOMENDADO:**
    1. **Certifique-se** que está selecionado "WebRTC (Tempo Real)" na barra lateral
    2. **Clique em "START"** para ativar a camera
    3. **Permita o acesso à câmera** quando solicitado pelo navegador
    4. **Aponte a câmera** para objetos - a deteccao e CONTINUA e AUTOMATICA
    5. **Mova a camera** - os objetos sao detectados em tempo real
    6. **Clique em "STOP"** para parar a detecção
    
    **📷 Modo Upload de Imagem (Alternativo):**
    1. **Selecione "Upload de Imagem"** na barra lateral
    2. **Faça upload** de uma imagem (JPG, PNG, BMP)
    3. **Veja o resultado** com objetos detectados marcados
    """)
    
    st.markdown("---")
    
    if mode == "WebRTC (Tempo Real)":
        st.markdown("### 🎥 Deteccao em Tempo Real")
        st.markdown("**Camera ativa - Deteccao continua de objetos**")
        
        # Componente WebRTC simplificado
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            video_processor_factory=VideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Status da conexão
        if webrtc_ctx.state.playing:
            st.success("🎥 Camera ATIVA - Deteccao em tempo real funcionando!")
            st.info("Aponte a camera para objetos. A deteccao e continua e automatica.")
        else:
            st.info("📹 Clique em START para ativar a camera em tempo real")
            
            # Seção de troubleshooting
            with st.expander("Problemas com conexao? Clique aqui para solucoes"):
                st.markdown("""
                **Para funcionar em tempo real, siga estes passos:**
                
                1. **Use Chrome** (melhor compatibilidade com WebRTC)
                2. **Permita acesso à camera** quando solicitado
                3. **Feche outras abas** que usem a camera
                4. **Desative VPN/proxy** se estiver usando
                5. **Teste em modo incógnito** primeiro
                
                **Se ainda não funcionar:**
                - Recarregue a página (Ctrl+F5)
                - Reinicie o navegador
                - Teste em outro navegador
                - Verifique se a camera não está sendo usada por outro programa
                
                **Dica:** Se o problema persistir, use o modo "Upload de Imagem" como alternativa.
                """)
    
    else:  # Modo Upload de Imagem
        st.markdown("### Upload de Imagem para Deteccao")
        
        uploaded_file = st.file_uploader(
            "Escolha uma imagem",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Faça upload de uma imagem para detectar objetos"
        )
        
        if uploaded_file is not None:
            # Converter para OpenCV
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detectar objetos
            img_with_detections = detect_objects(img.copy(), net, confidence_threshold)
            
            # Converter para RGB para exibição
            img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
            
            # Exibir resultado
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Imagem Original**")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with col2:
                st.markdown("**Imagem com Deteccoes**")
                st.image(img_rgb, use_column_width=True)
            
            st.success("Deteccao concluida! Objetos identificados estão marcados com retângulos verdes.")
    
    # Estatísticas (se disponível)
    if mode == "WebRTC (Tempo Real)" and webrtc_ctx.video_transformer:
        st.markdown("---")
        st.markdown("### Informacoes Tecnicas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", "Ativo" if webrtc_ctx.state.playing else "Inativo")
        
        with col2:
            st.metric("Limiar de Confianca", f"{confidence_threshold:.1f}")
        
        with col3:
            st.metric("Modelo", "YOLOv4")

if __name__ == "__main__":
    main()

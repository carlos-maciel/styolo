# Styolo - Deteccao de Objetos

Aplicacao web para deteccao de objetos usando **Streamlit** e **YOLOv4**. Tire uma foto ou faca upload de uma imagem para identificar objetos.

## Sobre o Projeto

Esta aplicacao permite detectar objetos atraves da camera do seu dispositivo ou por upload de imagens. Utiliza YOLOv4 para processamento e Streamlit para interface web. Funciona em desktop e dispositivos moveis.

## Tecnologias

- **Python 3.8+**
- **Streamlit** - Interface web
- **OpenCV** - Processamento de imagem
- **YOLOv4** - Deteccao de objetos
- **NumPy** - Manipulacao de arrays
- **Pillow** - Processamento de imagens

## Deploy no Streamlit Cloud

Para usar diretamente no navegador:

1. Faca fork deste repositorio no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositorio
4. Deploy automatico

A aplicacao estara disponivel online sem necessidade de instalacao local.

## Instalacao Local

### Pre-requisitos

- Python 3.8 ou superior
- Conexao com internet (para download do modelo)
- Camera (para tirar fotos)

### Passo 1: Clonar o Repositorio

```bash
git clone https://github.com/SEU-USUARIO/styolo.git
cd styolo
```

### Passo 2: Criar Ambiente Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Passo 4: Baixar Modelo YOLO

```bash
python download_yolo.py
```

Tempo estimado: 5-10 minutos (arquivo de ~250MB)

### Passo 5: Executar

**Metodo Automatico:**
```bash
python run.py
```

**Metodo Manual:**
```bash
streamlit run app.py --server.address localhost --server.port 8501
```

### Passo 6: Acessar

Abra no navegador: http://localhost:8501

## Como Usar

### Modo Camera (Tirar Foto)

1. Selecione "Tirar Foto" na barra lateral
2. Permita acesso a camera quando solicitado
3. Enquadre o objeto e clique para capturar
4. Aguarde a deteccao automatica
5. Veja os resultados com objetos identificados

### Modo Upload

1. Selecione "Upload de Imagem" na barra lateral
2. Clique para escolher uma imagem (JPG, PNG, BMP)
3. Aguarde a deteccao automatica
4. Veja os resultados

## Configuracoes

### Limiar de Confianca

- **Padrao:** 0.5
- **Range:** 0.1 - 1.0
- **Valores baixos** (0.1-0.3): Detecta mais objetos, pode ter falsos positivos
- **Valores medios** (0.4-0.6): Equilibrio entre precisao e deteccao
- **Valores altos** (0.7-1.0): Mais preciso, pode perder alguns objetos

### Classes Detectaveis

O modelo YOLOv4 detecta 80 classes diferentes:

- **Pessoas e animais:** pessoa, gato, cachorro, cavalo, passaro, etc.
- **Veiculos:** carro, moto, onibus, caminhao, bicicleta, etc.
- **Objetos domesticos:** cadeira, mesa, TV, laptop, celular, etc.
- **Alimentos:** maca, banana, pizza, garrafa, copo, etc.

## Estrutura do Projeto

```
styolo/
├── app.py              # Aplicacao principal Streamlit
├── run.py              # Script de inicializacao automatica
├── download_yolo.py    # Script para download dos arquivos YOLO
├── requirements.txt    # Dependencias Python
├── README.md           # Este arquivo
├── GUIA_EXECUCAO.md    # Guia rapido de execucao
├── yolov4.weights      # Pesos do modelo (baixado automaticamente)
└── yolov4.cfg          # Configuracao do modelo (baixado automaticamente)
```

## Solucao de Problemas

### Erro: "Python nao encontrado"
Instale o Python e adicione ao PATH durante a instalacao.

### Erro: "pip nao encontrado"
```bash
python -m ensurepip --upgrade
```

### Erro: "Nao foi possivel carregar o modelo YOLO"
1. Verifique se os arquivos yolov4.weights e yolov4.cfg existem
2. Execute novamente: python download_yolo.py
3. Verifique sua conexao com internet

### Camera nao funciona
- Verifique permissoes do navegador para camera
- Teste em outro navegador (Chrome recomendado)
- Feche outros programas que usam a camera
- Em dispositivos moveis, use HTTPS

### Erro ao instalar dependencias
```bash
python -m pip install --upgrade pip
pip cache purge
pip install --user -r requirements.txt
```

## Requisitos do Sistema

### Minimos
- CPU: Intel i5 ou equivalente
- RAM: 4GB
- Python: 3.8+
- Espaco em disco: 500MB

### Recomendados
- CPU: Intel i7 ou equivalente
- RAM: 8GB+
- Python: 3.10+

## Funcionalidades

- Captura de foto pela camera do dispositivo
- Upload de imagens para analise
- 80 classes de objetos detectaveis
- Interface responsiva para desktop e mobile
- Configuracao de limiar de confianca
- Download automatico do modelo
- Cache do modelo para melhor performance

## Referencias

- [Documentacao Streamlit](https://docs.streamlit.io/)
- [Documentacao OpenCV](https://docs.opencv.org/)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)

## Licenca

Este projeto esta sob a licenca MIT.

# 🎯 Detecção de Objetos em Tempo Real

Aplicação web para detecção de objetos em tempo real usando **Streamlit**, **WebRTC** e **YOLOv4**.

## 📋 Sobre o Projeto

Esta aplicação permite detectar objetos em tempo real através da câmera do seu dispositivo, utilizando YOLOv4 para processamento de imagens e Streamlit para interface web.

## 🛠️ Tecnologias

- **Python 3.8+**
- **Streamlit** - Interface web
- **OpenCV** - Processamento de imagem
- **YOLOv4** - Detecção de objetos
- **WebRTC** - Captura de vídeo em tempo real
- **NumPy** - Manipulação de arrays
- **Pillow** - Processamento de imagens

## 📦 Instalação Passo a Passo

### Pré-requisitos

- Python 3.8 ou superior instalado
- Conexão com internet (para download das dependências e modelo)
- Câmera web (para detecção em tempo real)

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd projeto-jardel
```

### Passo 2: Verificar Instalação do Python

Abra o terminal e verifique a versão do Python:

```bash
python --version
```

**Resultado esperado:** Python 3.8.x ou superior

Se não tiver Python instalado:
- **Windows:** Baixe em [python.org](https://www.python.org/downloads/)
- **Linux:** `sudo apt install python3 python3-pip`
- **Mac:** `brew install python3`

### Passo 3: Criar Ambiente Virtual (Recomendado)

Criar um ambiente virtual isola as dependências do projeto:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Como saber se funcionou:** Você verá `(venv)` no início da linha do terminal.

### Passo 4: Atualizar o pip

```bash
python -m pip install --upgrade pip
```

### Passo 5: Instalar Dependências

```bash
pip install -r requirements.txt
```

**Tempo estimado:** 2-5 minutos dependendo da conexão

**Se der erro de permissão no Windows:**
```bash
pip install --user -r requirements.txt
```

**Se der erro de processo em uso:**
- Feche outros programas Python
- Feche o Streamlit se estiver rodando
- Tente novamente

### Passo 6: Baixar Arquivos do Modelo YOLO

```bash
python download_yolo.py
```

**Tempo estimado:** 5-10 minutos (arquivo de ~250MB)

**O que será baixado:**
- `yolov4.weights` (~250MB) - Pesos do modelo treinado
- `yolov4.cfg` - Configuração da arquitetura

### Passo 7: Executar a Aplicação

**Método Automático (Recomendado):**
```bash
python run.py
```

Este script:
- ✅ Verifica a versão do Python
- ✅ Verifica arquivos necessários
- ✅ Instala dependências automaticamente
- ✅ Baixa arquivos YOLO se necessário
- ✅ Inicia a aplicação

**Método Manual:**
```bash
streamlit run app.py --server.address localhost --server.port 8501
```

### Passo 8: Acessar a Aplicação

Após executar, o navegador abrirá automaticamente em:
- **URL:** http://localhost:8501

Se não abrir automaticamente, acesse manualmente no navegador.

## 🚀 Como Usar

### Modo Tempo Real (WebRTC)

1. Na barra lateral, certifique-se que está selecionado **"WebRTC (Tempo Real)"**
2. Clique no botão **"START"** para ativar a câmera
3. **Permita o acesso à câmera** quando o navegador solicitar
4. Aponte a câmera para objetos - a detecção é **automática e contínua**
5. Ajuste o **"Limiar de Confiança"** na barra lateral se necessário
6. Clique em **"STOP"** para parar

### Modo Upload de Imagem

1. Na barra lateral, selecione **"Upload de Imagem"**
2. Clique em **"Escolha uma imagem"** e selecione um arquivo (JPG, PNG, BMP)
3. A imagem será processada e os objetos detectados aparecerão marcados

## ⚙️ Configurações

### Limiar de Confiança

- **Padrão:** 0.5
- **Range:** 0.1 - 1.0
- **Como funciona:**
  - Valores **baixos** (0.1-0.3): Detecta mais objetos, mas pode ter falsos positivos
  - Valores **médios** (0.4-0.6): Equilíbrio entre precisão e detecção
  - Valores **altos** (0.7-1.0): Mais preciso, mas pode perder alguns objetos

### Classes Detectáveis

O modelo YOLOv4 detecta **80 classes** diferentes:
- **Pessoas e animais:** pessoa, gato, cachorro, cavalo, etc.
- **Veículos:** carro, moto, ônibus, caminhão, etc.
- **Objetos domésticos:** cadeira, mesa, TV, laptop, etc.
- **Alimentos:** maçã, banana, pizza, garrafa, etc.

## 🔧 Estrutura do Projeto

```
projeto-jardel/
├── app.py                 # Aplicação principal Streamlit
├── run.py                 # Script de inicialização automática
├── download_yolo.py       # Script para download dos arquivos YOLO
├── requirements.txt       # Dependências Python
├── README.md              # Este arquivo
├── GUIA_EXECUCAO.md      # Guia rápido de execução
├── yolov4.weights        # Pesos do modelo (baixado automaticamente)
└── yolov4.cfg            # Configuração do modelo (baixado automaticamente)
```

## 🐛 Solução de Problemas

### Erro: "Python não encontrado"
**Solução:** Instale o Python e adicione ao PATH durante a instalação

### Erro: "pip não encontrado"
**Solução:** 
```bash
python -m ensurepip --upgrade
```

### Erro: "Streamlit já está em uso"
**Solução:**
```bash
# Windows
taskkill /f /im streamlit.exe

# Linux/Mac
pkill -f streamlit
```

### Erro: "Não foi possível carregar o modelo YOLO"
**Solução:**
1. Verifique se os arquivos `yolov4.weights` e `yolov4.cfg` existem
2. Execute novamente: `python download_yolo.py`
3. Verifique sua conexão com internet

### Erro: "ERR_ADDRESS_INVALID" no navegador
**Solução:** Use `localhost` em vez de `0.0.0.0`:
```bash
streamlit run app.py --server.address localhost
```

### Câmera não funciona
**Solução:**
- Verifique permissões do navegador para câmera
- Teste em outro navegador (Chrome funciona melhor)
- Feche outros programas que usam a câmera
- Teste em modo incógnito

### Performance baixa
**Solução:**
- Aumente o limiar de confiança (menos processamento)
- Feche outros programas pesados
- Use um navegador mais leve
- Considere usar GPU se disponível

### Erro ao instalar dependências
**Solução:**
```bash
# Atualizar pip
python -m pip install --upgrade pip

# Limpar cache
pip cache purge

# Reinstalar
pip install --user -r requirements.txt
```

## 📊 Requisitos do Sistema

### Mínimos
- **CPU:** Intel i5 ou equivalente
- **RAM:** 4GB
- **Python:** 3.8+
- **Espaço em disco:** 500MB (para modelo e dependências)

### Recomendados
- **CPU:** Intel i7 ou equivalente
- **RAM:** 8GB+
- **GPU:** NVIDIA com CUDA (opcional, melhora performance)
- **Python:** 3.10+

## 🎯 Funcionalidades

- ✅ Detecção em tempo real via WebRTC
- ✅ Upload de imagens para análise
- ✅ 80 classes de objetos detectáveis
- ✅ Interface web responsiva
- ✅ Configuração de limiar de confiança
- ✅ Download automático do modelo
- ✅ Cache do modelo para melhor performance

## 📚 Referências

- [Documentação Streamlit](https://docs.streamlit.io/)
- [Documentação OpenCV](https://docs.opencv.org/)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc)

## 👨‍💻 Autor

**Jardel** - Professor da disciplina de Machine Learning e Visão Computacional
**Matheus** - Monitor disciplina de Machine Learning e Visão Computacional

## 📄 Licença

Este projeto está sob a licença MIT.

---

**🎯 Divirta-se detectando objetos!**

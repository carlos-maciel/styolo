# 🚀 Guia Rápido de Execução

## ⚡ Início Rápido (3 Passos)

### 1️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Baixar Modelo YOLO
```bash
python download_yolo.py
```

### 3️⃣ Executar Aplicação
```bash
python run.py
```

**Pronto!** A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

---

## 📝 Método Detalhado

### Opção A: Execução Automática (Recomendado)

O script `run.py` faz tudo automaticamente:

```bash
python run.py
```

**O que ele faz:**
- ✅ Verifica versão do Python
- ✅ Verifica arquivos necessários
- ✅ Instala dependências
- ✅ Baixa arquivos YOLO se necessário
- ✅ Inicia a aplicação

### Opção B: Execução Manual

Se preferir fazer passo a passo:

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar modelo YOLO
python download_yolo.py

# 3. Executar aplicação
streamlit run app.py --server.address localhost --server.port 8501
```

---

## 🌐 Acessar a Aplicação

Após executar, acesse no navegador:
- **URL Local:** http://localhost:8501
- **URL Alternativa:** http://127.0.0.1:8501

---

## 📱 Como Usar a Interface

### Modo Tempo Real (WebRTC)

1. Certifique-se que **"WebRTC (Tempo Real)"** está selecionado na barra lateral
2. Clique em **"START"** para ativar a câmera
3. **Permita acesso à câmera** quando solicitado
4. Aponte a câmera para objetos
5. A detecção é **automática e contínua**
6. Ajuste o **"Limiar de Confiança"** se necessário
7. Clique em **"STOP"** para parar

### Modo Upload de Imagem

1. Selecione **"Upload de Imagem"** na barra lateral
2. Clique em **"Escolha uma imagem"**
3. Selecione um arquivo (JPG, PNG, BMP)
4. Veja o resultado com objetos marcados

---

## ⚠️ Problemas Comuns

### ❌ "Streamlit já está em uso"
```bash
# Windows
taskkill /f /im streamlit.exe

# Linux/Mac
pkill -f streamlit
```

### ❌ "Modelo não encontrado"
```bash
python download_yolo.py
```

### ❌ "Erro ao instalar dependências"
```bash
pip install --upgrade pip
pip install --user -r requirements.txt
```

### ❌ "Câmera não funciona"
- Verifique permissões do navegador
- Teste em outro navegador (Chrome recomendado)
- Feche outros programas que usam a câmera

---

## 💡 Dicas

- **Primeira execução:** Pode demorar alguns minutos para baixar o modelo
- **Melhor navegador:** Chrome ou Edge (melhor suporte WebRTC)
- **Performance:** Aumente o limiar de confiança se estiver lento
- **Parar aplicação:** Pressione `Ctrl+C` no terminal

---

## 📖 Documentação Completa

Para mais detalhes, consulte o [README.md](README.md)
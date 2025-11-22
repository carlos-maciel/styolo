# Guia Rapido de Execucao

## Inicio Rapido (3 Passos)

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Baixar Modelo YOLO
```bash
python download_yolo.py
```

### 3. Executar Aplicacao
```bash
python run.py
```

Pronto! A aplicacao abrira automaticamente no navegador em http://localhost:8501

---

## Metodo Detalhado

### Opcao A: Execucao Automatica (Recomendado)

O script run.py faz tudo automaticamente:

```bash
python run.py
```

O que ele faz:
- Verifica versao do Python
- Verifica arquivos necessarios
- Instala dependencias
- Baixa arquivos YOLO se necessario
- Inicia a aplicacao

### Opcao B: Execucao Manual

Se preferir fazer passo a passo:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Baixar modelo YOLO
python download_yolo.py

# 3. Executar aplicacao
streamlit run app.py --server.address localhost --server.port 8501
```

---

## Acessar a Aplicacao

Apos executar, acesse no navegador:
- URL Local: http://localhost:8501
- URL Alternativa: http://127.0.0.1:8501

---

## Como Usar a Interface

### Modo Camera (Tirar Foto)

1. Selecione "Tirar Foto" na barra lateral
2. Permita acesso a camera quando solicitado pelo navegador
3. Enquadre o objeto que deseja detectar
4. Clique no botao de captura
5. Aguarde a deteccao (alguns segundos)
6. Veja os resultados com objetos marcados

### Modo Upload de Imagem

1. Selecione "Upload de Imagem" na barra lateral
2. Clique para escolher um arquivo
3. Selecione uma imagem (JPG, PNG, BMP)
4. Veja o resultado com objetos detectados

---

## Problemas Comuns

### "Streamlit ja esta em uso"
```bash
# Windows
taskkill /f /im streamlit.exe

# Linux/Mac
pkill -f streamlit
```

### "Modelo nao encontrado"
```bash
python download_yolo.py
```

### "Erro ao instalar dependencias"
```bash
pip install --upgrade pip
pip install --user -r requirements.txt
```

### "Camera nao funciona"
- Verifique permissoes do navegador
- Teste em outro navegador (Chrome recomendado)
- Feche outros programas que usam a camera
- Em mobile, certifique-se de usar HTTPS

---

## Dicas

- Primeira execucao: Pode demorar alguns minutos para baixar o modelo
- Melhor navegador: Chrome ou Edge
- Performance: Aumente o limiar de confianca se estiver lento
- Parar aplicacao: Pressione Ctrl+C no terminal
- Mobile: Funciona melhor em modo retrato

---

## Documentacao Completa

Para mais detalhes, consulte o [README.md](README.md)

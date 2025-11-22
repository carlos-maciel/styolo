#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    try:
        print(f"[EXECUTANDO] {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} - Concluido!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Erro em {description}: {e}")
        print(f"Saida de erro: {e.stderr}")
        return False

def kill_streamlit_processes():
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'streamlit' in proc.info['name'].lower():
                    print(f"[INFO] Encontrado processo Streamlit (PID: {proc.info['pid']}) - finalizando...")
                    proc.kill()
                    proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except ImportError:
        print("[AVISO] psutil não disponível - tentando método alternativo")
        try:
            subprocess.run("taskkill /f /im streamlit.exe", shell=True, capture_output=True)
        except:
            pass

def upgrade_pip():
    print("[INFO] Atualizando pip...")
    return run_command("python -m pip install --upgrade pip", "Atualização do pip")

def install_dependencies():
    print("[INFO] Verificando e corrigindo problemas de instalação...")
    
    kill_streamlit_processes()
    upgrade_pip()
    
    print("[INFO] Tentando instalação com --user...")
    if run_command("pip install --user -r requirements.txt", "Instalação das dependências (--user)"):
        return True
    
    print("[INFO] Tentando instalação normal...")
    if run_command("pip install -r requirements.txt", "Instalação das dependências"):
        return True
    
    print("[INFO] Tentando instalação individual dos pacotes...")
    packages = [
        "streamlit==1.50.0",
        "opencv-python-headless==4.12.0.88", 
        "streamlit-webrtc==0.63.11",
        "numpy==2.2.6",
        "Pillow==11.3.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install --user {package}", f"Instalação de {package}"):
            print(f"[AVISO] Falha ao instalar {package}, continuando...")
    
    return True

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERRO] Python 3.8+ e necessario. Versao atual:", f"{version.major}.{version.minor}")
        return False
    print(f"[OK] Python {version.major}.{version.minor} detectado")
    return True

def check_files():
    required_files = ['app.py', 'requirements.txt', 'download_yolo.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"[ERRO] Arquivos faltando: {', '.join(missing_files)}")
        return False
    
    print("[OK] Todos os arquivos necessarios estao presentes")
    return True

def main():
    print("Inicializacao do Projeto de Deteccao de Objetos")
    print("=" * 60)
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_files():
        sys.exit(1)
    
    print("\nInstalando dependencias...")
    if not install_dependencies():
        print("[ERRO] Falha na instalacao das dependencias")
        sys.exit(1)
    
    print("\nVerificando arquivos do modelo YOLO...")
    if not os.path.exists("yolov4.weights") or not os.path.exists("yolov4.cfg"):
        print("Baixando arquivos do modelo YOLO...")
        if not run_command("python download_yolo.py", "Download dos arquivos YOLO"):
            print("[ERRO] Falha no download dos arquivos YOLO")
            sys.exit(1)
    else:
        print("[OK] Arquivos do modelo YOLO ja estao presentes")
    
    print("\nIniciando aplicacao...")
    print("=" * 60)
    print("A aplicacao sera aberta no seu navegador")
    print("URL: http://localhost:8501")
    print("Para parar a aplicacao, pressione Ctrl+C")
    print("=" * 60)
    
    try:
        print("[INFO] Iniciando Streamlit com configuração localhost...")
        subprocess.run([
            "streamlit", "run", "app.py", 
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "true"
        ], check=True)
    except KeyboardInterrupt:
        print("\nAplicacao encerrada pelo usuario")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO] Erro ao executar a aplicacao: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
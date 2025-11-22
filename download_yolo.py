#!/usr/bin/env python3
"""
Script para download automatico dos arquivos do modelo YOLO
Este script baixa os arquivos necessarios para executar a deteccao de objetos
"""

import os
import urllib.request
import sys
from pathlib import Path

def download_file(url, filename, description=""):
    """Baixa um arquivo da URL especificada"""
    try:
        print(f"Baixando {description}...")
        print(f"URL: {url}")
        print(f"Arquivo: {filename}")
        
        # Criar barra de progresso simples
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                sys.stdout.write(f"\rProgresso: {percent}% ({downloaded // 1024 // 1024}MB/{total_size // 1024 // 1024}MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filename, show_progress)
        print(f"\n[OK] {description} baixado com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n[ERRO] Erro ao baixar {description}: {e}")
        return False

def main():
    """Funcao principal para download dos arquivos YOLO"""
    print("Download dos Arquivos do Modelo YOLO")
    print("=" * 50)
    
    # URLs dos arquivos
    files_to_download = [
        {
            "url": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
            "filename": "yolov4.weights",
            "description": "Arquivo de pesos do YOLOv4 (~245MB)"
        },
        {
            "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "filename": "yolov4.cfg",
            "description": "Arquivo de configuracao do YOLOv4"
        }
    ]
    
    # Verificar se os arquivos ja existem
    existing_files = []
    for file_info in files_to_download:
        if os.path.exists(file_info["filename"]):
            existing_files.append(file_info["filename"])
    
    if existing_files:
        print(f"Arquivos ja existentes: {', '.join(existing_files)}")
        response = input("Deseja baixar novamente? (s/N): ").lower()
        if response != 's':
            print("Download cancelado.")
            return
    
    # Baixar arquivos
    success_count = 0
    for file_info in files_to_download:
        if not os.path.exists(file_info["filename"]) or response == 's':
            if download_file(file_info["url"], file_info["filename"], file_info["description"]):
                success_count += 1
        else:
            print(f"[PULANDO] {file_info['filename']} (ja existe)")
            success_count += 1
    
    print("\n" + "=" * 50)
    if success_count == len(files_to_download):
        print("Todos os arquivos foram baixados com sucesso!")
        print("\nProximos passos:")
        print("1. Instale as dependencias: pip install -r requirements.txt")
        print("2. Execute a aplicacao: streamlit run app.py")
    else:
        print("Alguns arquivos nao foram baixados. Verifique sua conexao com a internet.")
        sys.exit(1)

if __name__ == "__main__":
    main()

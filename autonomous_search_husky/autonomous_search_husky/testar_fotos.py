import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os

# 1. Oculta a janela principal do Tkinter (para usar apenas o pop-up de seleção)
root = tk.Tk()
root.withdraw()

# 2. Abre a janela para o usuário selecionar uma ou múltiplas imagens
caminhos_imagens = filedialog.askopenfilenames(
    title="Selecione as fotos para testar o YOLO",
    filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
)

if not caminhos_imagens:
    print("Nenhuma imagem selecionada. Encerrando o script.")
    exit()

# 3. Carregue o seu modelo aqui. 
# Mude "best.pt" para o caminho do seu modelo treinado, ou use "yolov8n-pose.pt" / "yolov8n.pt"
caminho_modelo = "yolov8n-pose.pt" 
print(f"Carregando modelo: {caminho_modelo}...")

try:
    model = YOLO(caminho_modelo)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# 4. Processa cada imagem selecionada
for caminho in caminhos_imagens:
    nome_arquivo = os.path.basename(caminho)
    print(f"\nAnalisando: {nome_arquivo}")
    
    # Roda a predição (ajuste o conf se quiser que ele seja mais/menos sensível)
    results = model.predict(source=caminho, conf=0.50, show=False)
    
    # Gera a matriz de pixels com os retângulos/esqueletos desenhados
    imagem_anotada = results[0].plot()
    
    # Redimensiona a imagem para caber na tela caso a foto original seja muito grande (ex: 4K)
    altura, largura = imagem_anotada.shape[:2]
    limite_altura = 800
    if altura > limite_altura:
        escala = limite_altura / altura
        imagem_anotada = cv2.resize(imagem_anotada, (int(largura * escala), int(altura * escala)))

    # Exibe a imagem na tela
    janela_titulo = f"Teste YOLO: {nome_arquivo} (Aperte QUALQUER TECLA para a proxima)"
    cv2.imshow(janela_titulo, imagem_anotada)
    
    # Pausa a execução e aguarda o usuário apertar qualquer tecla (0 = infinito)
    cv2.waitKey(0)
    
    # Fecha a janela atual antes de abrir a próxima
    cv2.destroyWindow(janela_titulo)

cv2.destroyAllWindows()
print("\nTeste concluído.")
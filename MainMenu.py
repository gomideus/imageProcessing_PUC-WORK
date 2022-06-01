from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter import filedialog 
from PIL import ImageTk
import cv2
import os
from CalculoCaracteristicas import calcularCaracteristicas
from ClassificarTextura import *

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 500

SCREEN_CARACTERISTICAS_WIDTH = 300
SCREEN_CARACTERISTICAS_HEIGHT = 700

BUTTON_PADDING = 2

script_dir = os.path.dirname(__file__) # Pegar path do diretorio atual
rel_path = "cropped.png" # Nomeclatura padrão para image que foi recortada
pathImagemRecortada  = os.path.join(script_dir, rel_path)

# Para treino e classificacao
diretorioLido = False
classificadorTreinado = False

# Variaveis para recorte
cropping = False
cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0 # Coordenadas de recorte
image = None
oriImage = None
pathImagemNaTela = None
i = None

# Pegar pontos da janela selecionada com o mouse.
# Recortar, mostrar e salvar nova imagem
def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, pathImagemRecortada, cropped, i
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse movendo
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # Se soltar o botao do mouse
    elif event == cv2.EVENT_LBUTTONUP:
        # Gravar coordenadas
        x_end, y_end = x, y
        cropping = False # Finalizou o recorte
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: # Dois pontos encontrados
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imwrite("cropped.png",roi)
            cropped = True # Recorte já foi finalizado
            getImageFromPath(pathImagemRecortada) # Mostra imagem recortada na tela do TK

# Janela de recorte de imagem
def recorte():
    global cropped, i
    cropped = False
    if image is None:
        print("Sem imagem para recortar!")
        messagebox.showerror('Erro', 'Primeiro importe uma imagem!')
        return
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
    while not cropped:
        i = image.copy()
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        cv2.waitKey(1)
    cv2.destroyWindow("image")

# Para setar label com path da imagem mostrada na tela
def setPathLabel():
    global pathImagemNaTela, pathLabel
    var = StringVar()
    var = "Imagem carregada: {}".format(pathImagemNaTela)
    if len(var) > 0: 
        pathLabel.config(text = var, fg="green")

# Mostrar imagem na tela
def setImagemNaTela(imgFile):
    global image, oriImage, pathImagemNaTela
    pathImagemNaTela = imgFile
    image = cv2.imread(imgFile)
    oriImage = image.copy()
    setPathLabel()

# Pega a imagem a partir do path e chama função para mostrar imagem
def getImageFromPath(imagePath):
    if(len(imagePath) > 0):
        image = ImageTk.PhotoImage(file=imagePath)
        imagebox.config(image=image)
        imagebox.image = image
        setImagemNaTela(imagePath)

# Abre o explorador de arquivos para selecionar imagens PNG ou JPG
def pesquisarArquivos():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File", filetypes = (("Img files","*.png*"),("Img files","*.jpg*")))
    return filename

def setLabelDiretorio(dirPath):
    global dirLabel
    var = StringVar()
    var = "Diretório carregado: {}".format(dirPath)
    if len(var) > 0: 
        dirLabel.config(text = var, fg="green")

# Abre o explorador de arquivos para selecionar diretorio de treino/teste
def pesquisarDiretorio():
    global diretorioLido
    dirPath = filedialog.askdirectory()
    if len(dirPath) > 0:
        setDiretorioTreino(dirPath)
        setLabelDiretorio(dirPath)
        diretorioLido = True

# Fechar programa
def fecharMenu():
    res = messagebox.askquestion('sair', 'Tem certeza que deseja sair?')
    if res == 'yes':
        root.destroy()

def treinarClassificador():
    global classificadorTreinado, diretorioLido
    if diretorioLido:
        time = treinar()
        limited_time = round(time, 2)
        classificadorTreinado = True
        messagebox.showinfo('Sucesso', 'Classificador treinado com sucesso.\nTempo de execução: {} segundos.'.format(limited_time))
    else:
        messagebox.showerror('Erro', 'Primeiro leia um diretório!')

def disableButtons():
    mb["state"] = DISABLED
    crop_button["state"] = DISABLED

def enableButtons():
    mb["state"] = NORMAL
    crop_button["state"] = NORMAL

def classificarImagensRestantes():
    classificacaoLabel = Label( root, text=None, fg="red", font=0)
    classificacaoLabel.pack()
    classificacaoCorretaLabel = Label( root, text=None, fg="green", font=0)
    classificacaoCorretaLabel.pack()
    if classificadorTreinado:
        disableButtons()
        var = tk.IntVar()
        i = tk.IntVar()
        nextButton = Button(root, text="Próximo", height= 1, width=15, command=lambda: var.set(1))
        nextButton.pack(pady=3)
        files, classificacaoCorreta, predicts, execTime, matrizDeConfusao, sensibilidade, especificidade, precisao = classificar()
        limitedExecTime = round(execTime, 2)
        print("Tempo de exeucao: {} segundos.".format(limitedExecTime))
        botaoFinalizar = Button(root, text="Finalizar", height= 1, width=15, command= lambda: 
            [ 
            classificacaoLabel.destroy(),
            classificacaoCorretaLabel.destroy(),
            botaoFinalizar.destroy(),
            enableButtons(),
            abrirJanelaDadosClassificacao(execTime, matrizDeConfusao, sensibilidade, especificidade, precisao),
            nextButton.destroy()
            ])
        botaoFinalizar.pack()
        i=0
        j=0
        while i < len(files)-3:
            stringVar = tk.StringVar()
            stringVar = "Classificação: {}".format(files[i+1])
            if predicts[j] == classificacaoCorreta[j]:
                classificacaoLabel.config(text = stringVar, fg="green", font=3)
            else:
                classificacaoLabel.config(text = stringVar, fg="red", font=3)
            stringVar = "Classificação correta: {}".format(classificacaoCorreta[j])
            classificacaoCorretaLabel.config(text = stringVar, fg="blue", font=3)
            getImageFromPath(files[i])
            nextButton.wait_variable(var)
            i+=3
            j+=1
        if i == len(files)-3:
            nextButton.destroy(),
    else:
        messagebox.showerror('Erro', 'Primeiro realize o treino!')

def classificarImagem():
    if classificadorTreinado:
        prediction = classificarUmaImagem(pathImagemNaTela)
        messagebox.showinfo('CLASSIFICAÇÃO', 'Classificação BIRADS: {}'.format(prediction))
    else:
        messagebox.showerror('Erro', 'Primeiro realize o treino!')

# Chama função para cálculo de características da imagem na tela
def arquivoCaracteristicas():
    if image is None:
        print("Sem imagem para classificar!")
        messagebox.showerror('Erro', 'Primeiro importe uma imagem!')
    else:
        feats, entropy, execTime = calcularCaracteristicas(pathImagemNaTela)
        abrirJanelaCaracteristicas(feats, entropy, execTime)

# Abre janela e mostra características calculadas da imagem na tela
def abrirJanelaCaracteristicas(feats, entropy, execTime):
     
    newWindow = Toplevel(root)
    newWindow.title("Caracteristicas da imagem")
    newWindow.geometry("{}x{}".format(SCREEN_CARACTERISTICAS_HEIGHT, SCREEN_CARACTERISTICAS_WIDTH))

    Label(newWindow, text="CARACTERÍSTICAS",
            fg='#f00', pady=10, padx=10, font=13).pack()
    Label(newWindow,
          text = "Entropia: {:.4f}".format(entropy), font=3).pack()
    Label(newWindow,
          text = "Tempo de execução: {:.4f}s".format(execTime), font=3).pack()
    Label(newWindow,
          text = "Homogeneidade, energia: \n{}".format(feats), font=3).pack()

# Abre janela e mostra dados de classificação
def abrirJanelaDadosClassificacao(execTime, matrizDeConfusao, sensibilidade, especificidade, precisao):
     
    newWindow = Toplevel(root)
    newWindow.title("Classificação")
    newWindow.geometry("{}x{}".format(SCREEN_CARACTERISTICAS_HEIGHT, SCREEN_CARACTERISTICAS_WIDTH))

    Label(newWindow, text="CARACTERÍSTICAS | CLASSIFICAÇÃO",
            fg='#f00', pady=10, padx=10, font=13).pack()
    Label(newWindow,
          text = "Sensibilidade: {}".format(sensibilidade), font=3).pack()
    Label(newWindow,
          text = "Especificidade: {}".format(especificidade), font=3).pack()
    Label(newWindow,
          text = "Precisao: {}".format(precisao), font=3).pack()
    Label(newWindow,
          text = "Tempo de execução: {:.4f}s".format(execTime), font=3).pack()
    Label(newWindow,
          text = "Matriz de confusao: \n{}".format(matrizDeConfusao), font=3).pack()

root = tk.Tk()
root.title('Processamento de Imagens - Trabalho')
root.geometry("{}x{}".format(SCREEN_HEIGHT, SCREEN_WIDTH))

frame = tk.Frame(root)
frame.pack()

mb =  Menubutton ( root, text="OPCOES", relief=RAISED, height= 1, width=17 )
mb.pack(pady=BUTTON_PADDING)
mb.menu =  Menu ( mb, tearoff = 0 )
mb["menu"] =  mb.menu

# Abrir e visualizar uma imagem
mb.menu.add_command(label="Importar imagem", command=lambda: getImageFromPath(pesquisarArquivos()))

# Ler o diretório de imagens de treino/teste
mb.menu.add_command(label="Ler diretório treino/teste", command= pesquisarDiretorio)

# Treinar o classificador
mb.menu.add_command(label="Treinar classificador", command= treinarClassificador)

# Para testar 25% das imagens restantes
mb.menu.add_command(label="Testar imagens restantes", command= classificarImagensRestantes)

# Calcular e exibir as características para a imagem visualizada ou área selecionada
mb.menu.add_command(label="Calcular características", command= arquivoCaracteristicas)

# Classificar a imagem atual (importada ou recortada)
mb.menu.add_command(label="Classificar imagem", command= classificarImagem)

# botao para recortar imagem
crop_button = Button(root, text="Recortar Imagem", height= 1, width=15, command=recorte)
crop_button.pack()

# Botao para fechar
exit_button = Button(root, text="Sair", height= 1, width=15, command=fecharMenu)
exit_button.pack(pady=BUTTON_PADDING)

# Label de path da imagem
pathLabel = Label( root, text="Imagem carregada: N/A", fg="red")
pathLabel.place(relx = 0.0, rely = 1.0, anchor ='sw')

dirLabel = Label(root, text="Diretório carregado: N/A", fg="red")
dirLabel.place(relx = 0.0, rely = 0.95, anchor ='sw')

imagebox = tk.Label(root)
imagebox.pack()

root.mainloop()

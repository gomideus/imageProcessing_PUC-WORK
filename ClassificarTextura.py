import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
import random
import time
from sklearn.metrics import confusion_matrix

# Função para extrair textura de Haralick de uma imagem
def extract_features(image):
    # calcular recursos de textura haralick para 4 tipos de adjacência
    textures = mt.features.haralick(image)

    # Tomar a média disso e devolver
    ht_mean = textures.mean(axis = 0).reshape(1, -1)
    return ht_mean

# Carregar o conjunto de dados de treinamento
script_dir = os.path.dirname(__file__) # <-- Diretório absoluto em que o script está sendo executado
rel_path = "Imagens"
train_path  = os.path.join(script_dir, rel_path)
train_names = os.listdir(train_path)

def setDiretorioTreino(dirPath):
    global train_path, train_names
    train_path  = dirPath
    train_names = os.listdir(train_path)

# Listas vazias para conter vetores de recursos e rótulos de treino
train_features = []
train_labels   = []

paths = []
pathsTreinar = []
pathsTestar = []
classificacaoCorreta = []

matrizDeConfusao = []

def treinar(): 

    start_time = time.time()

    global paths, pathsTreinar, pathsTestar, classificacaoCorreta
    paths = []
    pathsTreinar = []
    pathsTestar = []
    classificacaoCorreta = []
    # Loop sobre o conjunto de dados treino
    for train_name in train_names:
        # cur_path = train_path + "//" + train_name
        cur_path = os.path.join(train_path, train_name)
        cur_label = train_name
        
        i = 0
        paths = []
        for file in glob.glob(os.path.join(cur_path, "*.png")): # Contar imagens do diretório e salvar caminhos
            i += 1
            paths.append(file)

        percentAmmount =  (int) ( (i*75)/100 ) # Calcular 75% das imagens
        control = 0
        while control < percentAmmount: # Selecionar 75% das imagens aleatoriamente para treino

            randomIndex = random.randint(0,i-1) # Índice de randomicidade

            if not ( paths[randomIndex] in pathsTreinar ): # Se path não tiver sido selecionado
                pathsTreinar.append(paths[randomIndex]) # Adicionar path de imagem imagem para treino
                control += 1

        j=0
        for file in glob.glob(os.path.join(cur_path, "*.png")):

            if file in pathsTreinar:

                image = cv2.imread(file)

                # Converter para tons de cinza
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Extrair textura de Haralick da imagem
                features = extract_features(gray)

                # Append vetor de feature e rótulo
                train_features.append(features.reshape(1, -1)[0])
                train_labels.append(cur_label)

                j+=1
        print("{} imagens processadas no diretório {}".format(j, train_name))

        # Selecionar o restante das imagens para teste
        for file in glob.glob(os.path.join(cur_path, "*.png")):
            if not ( file in pathsTreinar ):
                pathsTestar.append(file)
                classificacaoCorreta.append(cur_label)

    end = time.time() - start_time
    return end

    # Printando para dar uma olhada no tamanho do nosso vetor de recursos e rótulos
    # print ("Training features: {}".format(np.array(train_features).shape))
    # print ("Training labels: {}".format(np.array(train_labels).shape))

def calculoSensitivitySpecificity():
        FP = matrizDeConfusao.sum(axis=0) - np.diag(matrizDeConfusao)
        FN = matrizDeConfusao.sum(axis=1) - np.diag(matrizDeConfusao)
        TP = np.diag(matrizDeConfusao)
        TN = matrizDeConfusao.sum() - (FP + FN + TP)
        TPR = TP / (TP + FN) 
        TNR = TN / (TN + FP) 
        # ACC = (TP + TN) / (TP + FP + FN + TN)  # Overall accuracy
        sensitivity = round(TPR.mean(), 2)
        specificity = round(TNR.mean(), 2)
        accuracy = TP.sum() / matrizDeConfusao.sum()
        print("Sensibilidade: {}".format(sensitivity))
        print("Especificidade: {}".format(specificity))
        print("Accuracy: {}".format(accuracy))
        return sensitivity, specificity, accuracy

predicts = []
def classificar():

    global predicts, matrizDeConfusao
    predicts = []

    start_time = time.time()

    # Criando classificador
    clf_svm = LinearSVC(random_state = 9, dual=False)

    # Ajustar os dados e rótulos de treinamento
    clf_svm.fit(train_features, train_labels)

    # Loop sobre imagens teste
    # t_path = "Imagens//4"
    t_path = os.path.join("Imagens", "4")
    test_path = os.path.join(script_dir, t_path)
    files = []
    i = 0
    acertos = 0
    for file in pathsTestar: 
        # Ler imagem
        image = cv2.imread(file)

        # Converter para tons de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extrair textura de Haralick da imagem
        features = extract_features(gray)

        # Avaliar o modelo e prever o rótulo
        prediction = clf_svm.predict(features)

        files.append(file)
        files.append(prediction)
        files.append(test_path)

        for value in prediction:
            predictionFormatada = value

        predicts.append(predictionFormatada)

        if( classificacaoCorreta[i] == predictionFormatada ):
            acertos+=1

        print("Prevendo imagem: {}".format(file))
        print("Previsão = {} Previsão correta: {}\n".format(predictionFormatada, classificacaoCorreta[i]))
        i+=1
    percentualAcertos = (acertos/(i))*100
    limited_percent = round(percentualAcertos, 2)
    print("Acertos: {}/{} ({}%)".format(acertos, i, limited_percent))

    matrizDeConfusao = confusion_matrix(classificacaoCorreta, predicts)
    print(matrizDeConfusao)

    sensibilidade, especificidade, precisao = calculoSensitivitySpecificity()

    end = time.time() - start_time

    return files, classificacaoCorreta, predicts, end, matrizDeConfusao, sensibilidade, especificidade, precisao

def classificarUmaImagem(file):

    #Criar classificador
    clf_svm = LinearSVC(random_state = 9, dual=False)
    clf_svm.fit(train_features, train_labels)
    image = cv2.imread(file)

    # Converter para tons de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extrair textura de Haralick da imagem
    features = extract_features(gray)

    # Avaliar o modelo e prever o rótulo
    prediction = clf_svm.predict(features)
    return prediction

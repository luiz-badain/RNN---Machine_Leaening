
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Função para ler dados de um arquivo txt
def ler_dados_txt(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as file:
        linhas = file.readlines()
    corpus = [linha.strip() for linha in linhas if linha.strip()]
    return corpus

# Caminho do arquivo txt
caminho_arquivo = 'dados.txt'

# Ler o corpus do arquivo txt
corpus = ler_dados_txt(caminho_arquivo)

# Preparação dos dados
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Criar pares de entrada e saída
input_sequences = []
target_words = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence[:-1])
        target_words.append(n_gram_sequence[-1])

# Padronizar o comprimento das sequências de entrada
max_seq_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')

input_sequences = np.array(input_sequences)
target_words = np.array(target_words)

# Criar o modelo RNN
modelo = Sequential()
modelo.add(Embedding(vocab_size, 10, input_length=max_seq_length))
modelo.add(SimpleRNN(50, activation='relu'))
modelo.add(Dense(vocab_size, activation='softmax'))

# Compilar o modelo
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
modelo.fit(input_sequences, target_words, epochs=100)

# Função para prever a próxima palavra
def prever_proxima_palavra(modelo, tokenizer, texto, num_sugestoes=3):
    sequence = tokenizer.texts_to_sequences([texto])[0]
    sequence = pad_sequences([sequence], maxlen=max_seq_length, padding='pre')
    predictions = modelo.predict(sequence)
    sorted_indices = np.argsort(predictions[0])[-num_sugestoes:][::-1]
    sugestoes = [word for word, index in tokenizer.word_index.items() if index in sorted_indices]
    return sugestoes

# Função para construir a frase interativamente
def construir_frase_interativamente(modelo, tokenizer, texto_inicial, max_len=20):
    texto_atual = texto_inicial
    while len(texto_atual.split()) < max_len:
        sugestoes = prever_proxima_palavra(modelo, tokenizer, texto_atual, num_sugestoes=3)
        if len(sugestoes) == 0:
            break
        print(f'Sugestões: {", ".join(sugestoes)}')
        user_input = input("Digite a próxima palavra ou escolha uma das sugestões, e pressione '0' caso queira finalizar: ").strip()
        if user_input == '0':
            break
        if user_input and user_input in sugestoes:
            texto_atual += ' ' + user_input
        elif user_input:
            texto_atual += ' ' + user_input
        else:
            texto_atual += ' ' + sugestoes[0]
    return texto_atual

# Solicitar a frase inicial ao usuário
texto_inicial = input("Digite uma frase inicial: ")
frase_completa = construir_frase_interativamente(modelo, tokenizer, texto_inicial)

print(f'Frase completa: "{frase_completa}"')
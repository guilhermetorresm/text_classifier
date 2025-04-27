import pandas as pd
import os

# Definindo o caminho dos arquivos
data_dir = 'ag_news_csv'
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')

# Mapeamento das classes (o dataset usa classes de 1 a 4)
class_names = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}

# Lendo os arquivos CSV de treino e teste
train_df = pd.read_csv(train_path, header=None, names=['Class', 'Title', 'Description'])
test_df = pd.read_csv(test_path, header=None, names=['Class', 'Title', 'Description'])

# Adicionando o nome das classes
train_df['ClassName'] = train_df['Class'].map(class_names)
test_df['ClassName'] = test_df['Class'].map(class_names)

# Mostrando informações sobre os datasets
print("\n=== Dataset de Treino ===")
print(f"Formato: {train_df.shape}")
print("\nPrimeiras linhas:")
print(train_df.head())
print("\nDistribuição das classes:")
print(train_df['ClassName'].value_counts())

print("\n\n=== Dataset de Teste ===")
print(f"Formato: {test_df.shape}")
print("\nPrimeiras linhas:")
print(test_df.head())
print("\nDistribuição das classes:")
print(test_df['ClassName'].value_counts())

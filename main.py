# Classificação de Texto - AG's News Dataset
# Implementação da Tarefa 07 - Tópicos Especiais: classificação de texto

# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE
from collections import Counter
from transformers import RobertaModel, RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import sys
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')

# Configuração para exibição de gráficos e saídas
plt.style.use('ggplot')
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

# Função para carregar os dados
def load_data(train_path, test_path, sample_size=1.0):
    """
    Carrega os dados de treino e teste e realiza o pré-processamento inicial

    Parameters:
    train_path : str
        Caminho para o arquivo de treino
    test_path : str
        Caminho para o arquivo de teste
    sample_size : float, default=1.0
        Porcentagem de dados a serem carregados (entre 0.0 e 1.0)
    """
    print("=== Carregando os dados ===")
    print(f"Tamanho da amostra: {sample_size*100:.1f}%")
    
    # Verificar se o sample_size está dentro do intervalo válido
    if not 0.0 <= sample_size <= 1.0:
        raise ValueError("O tamanho da amostra deve estar entre 0.0 e 1.0")
    
    # Carregar os arquivos
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    # Aplicar amostragem se sample_size < 1.0
    if sample_size < 1.0:
        train_df = train_df.sample(frac=sample_size, random_state=42)
        test_df = test_df.sample(frac=sample_size, random_state=42)
    
    # Renomear colunas
    columns = ['Class', 'Title', 'Description']
    train_df.columns = columns
    test_df.columns = columns
    
    # Converter índices de classe para valores começando em 0 (para compatibilidade com algoritmos)
    train_df['Class'] = train_df['Class'] - 1
    test_df['Class'] = test_df['Class'] - 1
    
    # Mapear classes numéricas para nomes
    class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    train_df['ClassName'] = train_df['Class'].map(class_names)
    test_df['ClassName'] = test_df['Class'].map(class_names)
    
    # Combinar título e descrição em um único campo de texto
    train_df['Text'] = train_df['Title'] + ' ' + train_df['Description']
    test_df['Text'] = test_df['Title'] + ' ' + test_df['Description']
    
    print(f"=== Dataset de Treino ===")
    print(f"Formato: {train_df.shape}")
    print(f"Primeiras linhas:")
    print(train_df[['Class', 'Title', 'Description', 'ClassName']].head())
    print(f"Distribuição das classes:")
    print(train_df['ClassName'].value_counts())
    
    print(f"\n=== Dataset de Teste ===")
    print(f"Formato: {test_df.shape}")
    print(f"Primeiras linhas:")
    print(test_df[['Class', 'Title', 'Description', 'ClassName']].head())
    print(f"Distribuição das classes:")
    print(test_df['ClassName'].value_counts())
    
    return train_df, test_df

# Função para pré-processamento de texto
def preprocess_text(df):
    """
    Realiza o pré-processamento do texto nos dados
    """
    print("\n=== Pré-processamento do texto ===")
    start_time = time.time()
    
    # Download de recursos do NLTK necessários
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('omw-1.4')
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        # Converter para minúsculas
        text = text.lower()
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenização
        tokens = nltk.word_tokenize(text)
        # Remover stopwords e aplicar lemmatização
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    # Aplicar a função de limpeza na coluna 'Text'
    df['CleanText'] = df['Text'].apply(clean_text)
    
    # Mostrar exemplos de texto original e limpo
    print("Exemplos de texto antes e depois da limpeza:")
    for i in range(3):
        print(f"\nExemplo {i+1}:")
        print(f"Original: {df['Text'].iloc[i][:100]}...")
        print(f"Limpo: {df['CleanText'].iloc[i][:100]}...")
    
    print(f"\nTempo de pré-processamento: {time.time() - start_time:.2f} segundos")
    return df

# Função para criar embeddings com Word2Vec
def create_embeddings(train_df, test_df):
    """
    Cria representações vetoriais usando RoBERTa embeddings
    """
    print("\n=== Criando embeddings com RoBERTa ===")
    start_time = time.time()
    
    # Carregar modelo e tokenizador do RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    
    # Definir device (GPU se disponível, senão CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # Modo de avaliação
    
    # Função para extrair embeddings de um texto
    def get_roberta_embeddings(texts, batch_size=16):
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenizar os textos
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                     max_length=128, return_tensors='pt')
            
            # Mover para GPU se disponível
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            
            # Desativar cálculo de gradientes
            with torch.no_grad():
                # Obter embeddings do modelo
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Usar o embedding do token [CLS] (primeiro token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenar todos os batches
        return np.vstack(all_embeddings)
    
    # Extrair embeddings para os conjuntos de treino e teste
    print("Gerando embeddings RoBERTa para o conjunto de treino...")
    X_train_roberta = get_roberta_embeddings(train_df['CleanText'].tolist())
    
    print("Gerando embeddings RoBERTa para o conjunto de teste...")
    X_test_roberta = get_roberta_embeddings(test_df['CleanText'].tolist())
    
    # Normalizar os vetores
    X_train_roberta = normalize(X_train_roberta)
    X_test_roberta = normalize(X_test_roberta)
    
    print(f"Dimensões dos embeddings para treino: {X_train_roberta.shape}")
    print(f"Dimensões dos embeddings para teste: {X_test_roberta.shape}")
    print(f"Tempo de geração de embeddings: {time.time() - start_time:.2f} segundos")
    
    return X_train_roberta, X_test_roberta

# Função para criar vetorizações (Bag of Words, TF-IDF e Word Embeddings)
def create_vectorizations(train_df, test_df, results_dir):
    """
    Cria representações vetoriais usando BoW (Count), TF-IDF e Word Embeddings
    """
    print("\n=== Criando representações vetoriais ===")
    X_train = train_df['CleanText']
    y_train = train_df['Class']
    X_test = test_df['CleanText']
    y_test = test_df['Class']
    
    # Configuração do CountVectorizer (Bag of Words)
    print("Criando vetorização com Bag of Words (frequência de unigramas)...")
    start_time = time.time()
    count_vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 1))
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)
    print(f"Dimensões da matriz BoW para treino: {X_train_count.shape}")
    print(f"Dimensões da matriz BoW para teste: {X_test_count.shape}")
    print(f"Número de características (palavras únicas): {len(count_vectorizer.vocabulary_)}")
    print(f"Tempo de vetorização BoW: {time.time() - start_time:.2f} segundos")
    
    # Configuração do TfidfVectorizer
    print("\nCriando vetorização com TF-IDF...")
    start_time = time.time()
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print(f"Dimensões da matriz TF-IDF para treino: {X_train_tfidf.shape}")
    print(f"Dimensões da matriz TF-IDF para teste: {X_test_tfidf.shape}")
    print(f"Número de características (palavras únicas): {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Tempo de vetorização TF-IDF: {time.time() - start_time:.2f} segundos")
    
    # Criação de embeddings
    X_train_roberta, X_test_roberta = create_embeddings(train_df, test_df)
    
    # Visualização das palavras mais frequentes (top 20)
    feature_names = count_vectorizer.get_feature_names_out()
    word_counts = np.asarray(X_train_count.sum(axis=0)).ravel()
    word_freq = pd.DataFrame({'word': feature_names, 'count': word_counts})
    top_words = word_freq.sort_values('count', ascending=False).head(20)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='count', y='word', data=top_words)
    plt.title('Top 20 palavras mais frequentes no corpus')
    plt.xlabel('Frequência')
    plt.ylabel('Palavras')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'top_words.png'))
    plt.close()
    
    print("\nAs 20 palavras mais frequentes foram salvas como 'top_words.png'")
    
    return X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, X_train_roberta, X_test_roberta, y_train, y_test

# Função para avaliar um modelo
def evaluate_model(y_true, y_pred, class_names, model_name):
    """
    Avalia o modelo utilizando várias métricas
    """
    # Métricas gerais
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Métricas por classe
    class_precision = precision_score(y_true, y_pred, average=None)
    class_recall = recall_score(y_true, y_pred, average=None)
    class_f1 = f1_score(y_true, y_pred, average=None)
    
    # Criar e exibir o relatório de classificação
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Criar a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Retornar todas as métricas e visualizações
    results = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'report': report,
        'confusion_matrix': cm
    }
    
    return results

# Função para treinar e avaliar os modelos
def train_and_evaluate_models(X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, 
                             X_train_roberta, X_test_roberta, y_train, y_test):
    """
    Treina e avalia os modelos RF e SVM com diferentes representações, incluindo embeddings
    """
    print("\n=== Treinamento e avaliação dos modelos ===")
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    results = {}
    
    # Analisar desbalanceamento de classes
    class_counts = Counter(y_train)
    print("\nDistribuição das classes no conjunto de treino:")
    for class_idx, count in class_counts.items():
        print(f"Classe {class_idx} ({class_names[class_idx]}): {count} amostras")
    
    min_class = min(class_counts, key=class_counts.get)
    min_class_count = class_counts[min_class]
    print(f"Classe minoritária: {class_names[min_class]} com {min_class_count} amostras")
    
    # Aplica SMOTE apenas se houver desbalanceamento significativo
    if max(class_counts.values()) / min_class_count > 1.1:  # mais de 10% de diferença
        print("\nAplicando SMOTE para balancear classes...")
        start_time = time.time()
        
        smote = SMOTE(random_state=42)
        
        X_train_count_resampled, y_train_count_resampled = smote.fit_resample(X_train_count, y_train)
        X_train_tfidf_resampled, y_train_tfidf_resampled = smote.fit_resample(X_train_tfidf, y_train)
        X_train_roberta_resampled, y_train_roberta_resampled = smote.fit_resample(X_train_roberta, y_train)
        
        print(f"Tempo para aplicar SMOTE: {time.time() - start_time:.2f} segundos")
        
        # Verificar nova distribuição
        new_counts = Counter(y_train_count_resampled)
        print("Nova distribuição após SMOTE:")
        for class_idx, count in new_counts.items():
            print(f"Classe {class_idx} ({class_names[class_idx]}): {count} amostras")
    else:
        print("\nClasses já estão razoavelmente balanceadas, não é necessário aplicar SMOTE.")
        X_train_count_resampled, y_train_count_resampled = X_train_count, y_train
        X_train_tfidf_resampled, y_train_tfidf_resampled = X_train_tfidf, y_train
        X_train_roberta_resampled, y_train_roberta_resampled = X_train_roberta, y_train
    
    # 1. Random Forest com BoW
    print("\n1. Treinando Random Forest com Bag of Words...")
    start_time = time.time()
    rf_bow = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_bow.fit(X_train_count_resampled, y_train_count_resampled)
    y_pred_rf_bow = rf_bow.predict(X_test_count)
    train_time_rf_bow = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_rf_bow:.2f} segundos")
    
    results['RF_BoW'] = evaluate_model(y_test, y_pred_rf_bow, class_names, "Random Forest (BoW)")
    results['RF_BoW']['train_time'] = train_time_rf_bow
    
    # 2. Random Forest com TF-IDF
    print("\n2. Treinando Random Forest com TF-IDF...")
    start_time = time.time()
    rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_tfidf.fit(X_train_tfidf_resampled, y_train_tfidf_resampled)
    y_pred_rf_tfidf = rf_tfidf.predict(X_test_tfidf)
    train_time_rf_tfidf = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_rf_tfidf:.2f} segundos")
    
    results['RF_TF-IDF'] = evaluate_model(y_test, y_pred_rf_tfidf, class_names, "Random Forest (TF-IDF)")
    results['RF_TF-IDF']['train_time'] = train_time_rf_tfidf
    
    # 3. Random Forest com Embeddings
    print("\n3. Treinando Random Forest com Word Embeddings...")
    start_time = time.time()
    rf_roberta = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_roberta.fit(X_train_roberta_resampled, y_train_roberta_resampled)
    y_pred_rf_roberta = rf_roberta.predict(X_test_roberta)
    train_time_rf_roberta = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_rf_roberta:.2f} segundos")
    
    results['RF_ROBERTA'] = evaluate_model(y_test, y_pred_rf_roberta, class_names, "Random Forest (ROBERTA)")
    results['RF_ROBERTA']['train_time'] = train_time_rf_roberta
    
    # 4. SVM com BoW
    print("\n4. Treinando SVM com Bag of Words...")
    start_time = time.time()
    svm_bow = SVC(kernel='linear', random_state=42)
    svm_bow.fit(X_train_count_resampled, y_train_count_resampled)
    y_pred_svm_bow = svm_bow.predict(X_test_count)
    train_time_svm_bow = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_svm_bow:.2f} segundos")
    
    results['SVM_BoW'] = evaluate_model(y_test, y_pred_svm_bow, class_names, "SVM (BoW)")
    results['SVM_BoW']['train_time'] = train_time_svm_bow
    
    # 5. SVM com TF-IDF
    print("\n5. Treinando SVM com TF-IDF...")
    start_time = time.time()
    svm_tfidf = SVC(kernel='linear', random_state=42)
    svm_tfidf.fit(X_train_tfidf_resampled, y_train_tfidf_resampled)
    y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)
    train_time_svm_tfidf = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_svm_tfidf:.2f} segundos")
    
    results['SVM_TF-IDF'] = evaluate_model(y_test, y_pred_svm_tfidf, class_names, "SVM (TF-IDF)")
    results['SVM_TF-IDF']['train_time'] = train_time_svm_tfidf
    
    # 6. SVM com Embeddings
    print("\n6. Treinando SVM com Word Embeddings...")
    start_time = time.time()
    svm_roberta = SVC(kernel='linear', random_state=42)
    svm_roberta.fit(X_train_roberta_resampled, y_train_roberta_resampled)
    y_pred_svm_roberta = svm_roberta.predict(X_test_roberta)
    train_time_svm_roberta = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_svm_roberta:.2f} segundos")
    
    results['SVM_ROBERTA'] = evaluate_model(y_test, y_pred_svm_roberta, class_names, "SVM (ROBERTA)")
    results['SVM_ROBERTA']['train_time'] = train_time_svm_roberta
    
    return results

# Função para visualizar os resultados
def visualize_results(results, results_dir):
    """
    Cria e salva visualizações dos resultados dos modelos
    """
    print("\n=== Visualização e comparação dos resultados ===")
    
    # 1. Comparação de acurácia entre os modelos
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['lightblue', 'skyblue', 'lightgreen', 'darkgreen'])
    
    # Adicionar rótulos de valor em cada barra
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Comparação de Acurácia entre os Modelos')
    plt.xlabel('Modelo')
    plt.ylabel('Acurácia')
    plt.ylim(0.85, 1.0)  # Ajustar conforme necessário
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. Comparação de métricas macro para cada modelo
    metrics = ['macro_precision', 'macro_recall', 'macro_f1']
    metric_names = ['Precisão Macro', 'Revocação Macro', 'F1 Macro']
    
    metric_values = {model: [results[model][metric] for metric in metrics] for model in model_names}
    
    x = np.arange(len(metrics))
    width = 0.1
    
    plt.figure(figsize=(19, 10))
    
    # Plotar barras para cada modelo
    for i, (model, values) in enumerate(metric_values.items()):
        offset = width * (i - 1.5)
        bars = plt.bar(x + offset, values, width, label=model)
        
        # Adicionar rótulos de valor em cada barra
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.title('Comparação de Métricas Macro entre os Modelos')
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'macro_metrics_comparison.png'))
    plt.close()
    
    # 3. Métricas por classe para o melhor modelo (determinado pela acurácia)
    best_model = model_names[np.argmax(accuracies)]
    print(f"\nO melhor modelo é: {best_model} com acurácia de {results[best_model]['accuracy']:.4f}")
    
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    class_metrics = pd.DataFrame({
        'Classe': class_names,
        'Precisão': results[best_model]['class_precision'],
        'Revocação': results[best_model]['class_recall'],
        'F1': results[best_model]['class_f1']
    })
    
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['Precisão', 'Revocação', 'F1']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        bars = plt.bar(class_names, class_metrics[metric], color='skyblue')
        
        # Adicionar rótulos de valor em cada barra
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.title(f'{metric} por Classe para {best_model}')
        plt.ylim(0.8, 1.0)  # Ajustar conforme necessário
        plt.ylabel(metric)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'best_model_class_metrics.png'))
    plt.close()
    
    # 4. Matriz de confusão para cada modelo
    for model_name in model_names:
        cm = results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Prevista')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()
    
    # 5. Comparação do tempo de treinamento
    train_times = [results[model]['train_time'] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, train_times, color=['lightblue', 'skyblue', 'lightgreen', 'darkgreen'])
    
    # Adicionar rótulos de valor em cada barra
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}s', ha='center', va='bottom')
    
    plt.title('Comparação do Tempo de Treinamento entre os Modelos')
    plt.xlabel('Modelo')
    plt.ylabel('Tempo (segundos)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_time_comparison.png'))
    plt.close()
    
    # Imprimir resumo das métricas para todos os modelos
    print("\nResumo das métricas para todos os modelos:")
    metrics_summary = pd.DataFrame({
        'Modelo': model_names,
        'Acurácia': [results[model]['accuracy'] for model in model_names],
        'Precisão Macro': [results[model]['macro_precision'] for model in model_names],
        'Revocação Macro': [results[model]['macro_recall'] for model in model_names],
        'F1 Macro': [results[model]['macro_f1'] for model in model_names],
        'Tempo Treinamento (s)': [results[model]['train_time'] for model in model_names]
    })
    
    print(metrics_summary.to_string(index=False))
    return metrics_summary

# Função para gerar análise detalhada dos resultados
def detailed_analysis(results, train_df, test_df, results_dir):
    """
    Realiza uma análise detalhada dos resultados, focando especialmente nas classes minoritárias
    """
    print("\n=== Análise Detalhada dos Resultados ===")
    
    # Verificar a distribuição das classes
    class_dist_train = train_df['Class'].value_counts().sort_index()
    class_dist_test = test_df['Class'].value_counts().sort_index()
    
    print("\nDistribuição das classes no conjunto de treino:")
    print(class_dist_train)
    
    print("\nDistribuição das classes no conjunto de teste:")
    print(class_dist_test)
    
    # Identificar classe minoritária
    min_class_idx = class_dist_train.argmin()
    min_class_name = ['World', 'Sports', 'Business', 'Sci/Tech'][min_class_idx]
    
    print(f"\nClasse minoritária identificada: {min_class_name} (Índice {min_class_idx}) com {class_dist_train.min()} amostras")
    
    # Análise específica da classe minoritária
    print(f"\nAnálise detalhada da classe minoritária ({min_class_name}):")
    
    min_class_metrics = pd.DataFrame({
        'Modelo': [],
        'Precisão': [],
        'Revocação': [],
        'F1-Score': []
    })
    
    for model_name, model_results in results.items():
        precision = model_results['class_precision'][min_class_idx]
        recall = model_results['class_recall'][min_class_idx]
        f1 = model_results['class_f1'][min_class_idx]
        
        min_class_metrics = pd.concat([
            min_class_metrics,
            pd.DataFrame({
                'Modelo': [model_name],
                'Precisão': [precision],
                'Revocação': [recall],
                'F1-Score': [f1]
            })
        ], ignore_index=True)
    
    # Ordenar por F1-Score
    min_class_metrics = min_class_metrics.sort_values('F1-Score', ascending=False)
    print(min_class_metrics.to_string(index=False))
    
    # Visualizar o desempenho da classe minoritária
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    x = np.arange(len(min_class_metrics))
    
    # Plotar barras para cada métrica
    plt.bar(x - bar_width, min_class_metrics['Precisão'], bar_width, label='Precisão')
    plt.bar(x, min_class_metrics['Revocação'], bar_width, label='Revocação')
    plt.bar(x + bar_width, min_class_metrics['F1-Score'], bar_width, label='F1-Score')
    
    plt.xlabel('Modelos')
    plt.ylabel('Valor')
    plt.title(f'Desempenho dos modelos na classe minoritária ({min_class_name})')
    plt.xticks(x, min_class_metrics['Modelo'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'minority_class_performance.png'))
    plt.close()
    
    print(f"\nGráfico de desempenho para a classe minoritária salvo como 'minority_class_performance.png'")
    
    # Análise geral por classe para cada modelo
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    print("\nAnálise de desempenho por classe para cada modelo:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        class_f1 = model_results['class_f1']
        class_precision = model_results['class_precision']
        class_recall = model_results['class_recall']
        
        # Identificar a classe com menor desempenho
        worst_class_idx = np.argmin(class_f1)
        worst_class = class_names[worst_class_idx]
        
        print(f"Classe mais difícil: {worst_class}")
        print(f"F1-Score: {class_f1[worst_class_idx]:.4f}")
        print(f"Precisão: {class_precision[worst_class_idx]:.4f}")
        print(f"Revocação: {class_recall[worst_class_idx]:.4f}")
        
        # Identificar a classe com melhor desempenho
        best_class_idx = np.argmax(class_f1)
        best_class = class_names[best_class_idx]
        
        print(f"\nClasse mais fácil: {best_class}")
        print(f"F1-Score: {class_f1[best_class_idx]:.4f}")
        print(f"Precisão: {class_precision[best_class_idx]:.4f}")
        print(f"Revocação: {class_recall[best_class_idx]:.4f}")
    
    # Identificar o melhor modelo geral
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    best_model_idx = np.argmax(accuracies)
    best_model = model_names[best_model_idx]
    
    print(f"\nO melhor modelo geral é {best_model} com acurácia de {accuracies[best_model_idx]:.4f}")
    
    # Criar gráfico comparativo de F1 por classe para todos os modelos
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(class_names))
    width = 0.12  # Mais estreito para acomodar mais modelos
    
    for i, (model_name, model_results) in enumerate(results.items()):
        offset = width * (i - 2.5)
        plt.bar(x + offset, model_results['class_f1'], width, label=model_name)
    
    plt.xlabel('Classes')
    plt.ylabel('F1-Score')
    plt.title('Comparação de F1-Score por Classe e Modelo')
    plt.xticks(x, class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'f1_by_class_comparison.png'))
    plt.close()
    
    # Analisar os erros do melhor modelo
    best_model_cm = results[best_model]['confusion_matrix']
    print(f"\nAnálise dos tipos de erro para o melhor modelo ({best_model}):")
    
    # Calcular taxa de erro para cada classe
    for i, class_name in enumerate(class_names):
        total_samples = np.sum(best_model_cm[i, :])
        correct_predictions = best_model_cm[i, i]
        error_rate = 1 - (correct_predictions / total_samples)
        
        most_confused_with_idx = np.argmax(np.delete(best_model_cm[i, :], i))
        # Ajustar o índice se necessário
        if most_confused_with_idx >= i:
            most_confused_with_idx += 1
        
        most_confused_with = class_names[most_confused_with_idx]
        confusion_count = best_model_cm[i, most_confused_with_idx]
        
        print(f"\nClasse: {class_name}")
        print(f"Taxa de erro: {error_rate:.4f} ({total_samples - correct_predictions} de {total_samples})")
        print(f"Mais frequentemente confundida com: {most_confused_with} ({confusion_count} amostras)")
        
        # Análise adicional para a classe minoritária
        if i == min_class_idx:
            print(f"*** Esta é a classe minoritária! ***")
            
            # Análise específica de confusão da classe minoritária
            confusion_counts = best_model_cm[i, :]
            confusion_rates = confusion_counts / total_samples
            
            print(f"Distribuição de previsões para a classe minoritária:")
            for j, class_name_pred in enumerate(class_names):
                print(f"  -> {class_name_pred}: {confusion_counts[j]} amostras ({confusion_rates[j]*100:.2f}%)")
    
    print("\nAnálise dos resultados concluída e salva em arquivos de imagem.")
    
    # Análise comparativa dos diferentes vetorizadores para a classe minoritária
    plt.figure(figsize=(14, 6))
    
    # Agrupar modelos por tipo de vetorizador
    vectorizer_types = ['BoW', 'TF-IDF', 'ROBERTA']
    model_types = ['RF', 'SVM']
    
    # Criar dataframe para análise
    vectorizer_analysis = pd.DataFrame(columns=['Vetorizador', 'Modelo', 'F1-Score'])
    
    for model_name, model_results in results.items():
        model_type = model_name.split('_')[0]  # RF ou SVM
        vectorizer = model_name.split('_')[1]   # BoW, TF-IDF ou ROBERTA
        f1 = model_results['class_f1'][min_class_idx]
        
        vectorizer_analysis = pd.concat([
            vectorizer_analysis,
            pd.DataFrame({
                'Vetorizador': [vectorizer],
                'Modelo': [model_type],
                'F1-Score': [f1]
            })
        ], ignore_index=True)
    
    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Vetorizador', y='F1-Score', hue='Modelo', data=vectorizer_analysis)
    plt.title(f'Comparação de Vetorizadores para a Classe Minoritária ({min_class_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'vectorizer_comparison_minority.png'))
    plt.close()
    
    print(f"Comparação dos vetorizadores para a classe minoritária salva como 'vectorizer_comparison_minority.png'")
    
    return {
        'best_model': best_model,
        'best_accuracy': accuracies[best_model_idx],
        'minority_class': min_class_name,
        'minority_class_idx': min_class_idx
    }

# Função principal que executa o pipeline completo
def main():
    """
    Função principal que executa todo o pipeline de classificação de texto
    """
    print("=== Iniciando o pipeline de classificação de texto ===")
    start_time = time.time()
    
    # Criar a pasta results se não existir
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Configurar log para salvar saídas do terminal
    log_file = os.path.join(results_dir, 'execution_log.txt')
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            """
            Função principal que executa todo o pipeline de classificação de texto
            """
            print("=== Iniciando o pipeline de classificação de texto ===")
            start_time = time.time()
            
            # Caminhos para os conjuntos de dados
            train_path = 'ag_news_csv/train.csv'
            test_path = 'ag_news_csv/test.csv'
            
            # 1. Carregar os dados
            train_df, test_df = load_data(train_path, test_path, sample_size=0.01)
            
            # 2. Pré-processar o texto
            train_df = preprocess_text(train_df)
            test_df = preprocess_text(test_df)
            
            # 3. Criar vetorizações (Bag of Words, TF-IDF e Embeddings)
            X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, X_train_roberta, X_test_roberta, y_train, y_test = create_vectorizations(train_df, test_df, results_dir)
            
            # 4. Treinar e avaliar os modelos
            results = train_and_evaluate_models(X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, X_train_roberta, X_test_roberta, y_train, y_test)
            
            # 5. Visualizar os resultados
            metrics_summary = visualize_results(results, results_dir)
            
            # 6. Análise detalhada dos resultados
            analysis_results = detailed_analysis(results, train_df, test_df, results_dir)
            
            # Salvar resumo das métricas em CSV
            metrics_summary.to_csv(os.path.join(results_dir, 'model_metrics_summary.csv'), index=False)
            
            total_time = time.time() - start_time
            print(f"\n=== Pipeline de classificação concluído em {total_time/60:.2f} minutos ===")
            print(f"O melhor modelo foi {analysis_results['best_model']} com acurácia de {analysis_results['best_accuracy']:.4f}")
            print(f"A classe minoritária foi {analysis_results['minority_class']}")
            print("Todas as visualizações e métricas foram salvas em arquivos para análise posterior.")
            print(f"Log de execução salvo em: {log_file}")
            
if __name__ == "__main__":
    main()
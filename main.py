# Classificação de Texto - AG's News Dataset
# Implementação da Tarefa 07 - Tópicos Especiais: classificação de texto

# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Configuração para exibição de gráficos e saídas
plt.style.use('ggplot')
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

# Função para carregar os dados
def load_data(train_path, test_path):
    """
    Carrega os dados de treino e teste e realiza o pré-processamento inicial
    """
    print("=== Carregando os dados ===")
    
    # Carregar os arquivos
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
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

# Função para criar vetorizações (Bag of Words e TF-IDF)
def create_vectorizations(train_df, test_df):
    """
    Cria representações vetoriais usando BoW (Count) e TF-IDF
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
    plt.savefig('top_words.png')
    plt.close()
    
    print("\nAs 20 palavras mais frequentes foram salvas como 'top_words.png'")
    
    return X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, y_train, y_test

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
def train_and_evaluate_models(X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Treina e avalia os modelos RF e SVM com diferentes representações
    """
    print("\n=== Treinamento e avaliação dos modelos ===")
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    results = {}
    
    # 1. Random Forest com BoW
    print("\n1. Treinando Random Forest com Bag of Words...")
    start_time = time.time()
    rf_bow = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_bow.fit(X_train_count, y_train)
    y_pred_rf_bow = rf_bow.predict(X_test_count)
    train_time_rf_bow = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_rf_bow:.2f} segundos")
    
    results['RF_BoW'] = evaluate_model(y_test, y_pred_rf_bow, class_names, "Random Forest (BoW)")
    results['RF_BoW']['train_time'] = train_time_rf_bow
    
    # 2. Random Forest com TF-IDF
    print("\n2. Treinando Random Forest com TF-IDF...")
    start_time = time.time()
    rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_tfidf.fit(X_train_tfidf, y_train)
    y_pred_rf_tfidf = rf_tfidf.predict(X_test_tfidf)
    train_time_rf_tfidf = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_rf_tfidf:.2f} segundos")
    
    results['RF_TF-IDF'] = evaluate_model(y_test, y_pred_rf_tfidf, class_names, "Random Forest (TF-IDF)")
    results['RF_TF-IDF']['train_time'] = train_time_rf_tfidf
    
    # 3. SVM com BoW
    print("\n3. Treinando SVM com Bag of Words...")
    start_time = time.time()
    svm_bow = SVC(kernel='linear', random_state=42)
    svm_bow.fit(X_train_count, y_train)
    y_pred_svm_bow = svm_bow.predict(X_test_count)
    train_time_svm_bow = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_svm_bow:.2f} segundos")
    
    results['SVM_BoW'] = evaluate_model(y_test, y_pred_svm_bow, class_names, "SVM (BoW)")
    results['SVM_BoW']['train_time'] = train_time_svm_bow
    
    # 4. SVM com TF-IDF
    print("\n4. Treinando SVM com TF-IDF...")
    start_time = time.time()
    svm_tfidf = SVC(kernel='linear', random_state=42)
    svm_tfidf.fit(X_train_tfidf, y_train)
    y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)
    train_time_svm_tfidf = time.time() - start_time
    print(f"Tempo de treinamento: {train_time_svm_tfidf:.2f} segundos")
    
    results['SVM_TF-IDF'] = evaluate_model(y_test, y_pred_svm_tfidf, class_names, "SVM (TF-IDF)")
    results['SVM_TF-IDF']['train_time'] = train_time_svm_tfidf
    
    return results

# Função para visualizar os resultados
def visualize_results(results):
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
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    # 2. Comparação de métricas macro para cada modelo
    metrics = ['macro_precision', 'macro_recall', 'macro_f1']
    metric_names = ['Precisão Macro', 'Revocação Macro', 'F1 Macro']
    
    metric_values = {model: [results[model][metric] for metric in metrics] for model in model_names}
    
    x = np.arange(len(metrics))
    width = 0.2
    
    plt.figure(figsize=(12, 7))
    
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
    plt.savefig('macro_metrics_comparison.png')
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
    plt.savefig('best_model_class_metrics.png')
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
        plt.savefig(f'confusion_matrix_{model_name}.png')
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
    plt.savefig('training_time_comparison.png')
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
def detailed_analysis(results, train_df, test_df):
    """
    Realiza uma análise detalhada dos resultados, focando especialmente nas classes minoritárias
    """
    print("\n=== Análise Detalhada dos Resultados ===")
    
    # Verificar se existem classes minoritárias no conjunto de dados
    class_dist_train = train_df['Class'].value_counts()
    class_dist_test = test_df['Class'].value_counts()
    
    print("\nDistribuição das classes no conjunto de treino:")
    print(class_dist_train)
    
    print("\nDistribuição das classes no conjunto de teste:")
    print(class_dist_test)
    
    # Como o AG News é balanceado, vamos analisar qual classe é mais difícil de classificar
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
    width = 0.2
    
    for i, (model_name, model_results) in enumerate(results.items()):
        offset = width * (i - 1.5)
        plt.bar(x + offset, model_results['class_f1'], width, label=model_name)
    
    plt.xlabel('Classes')
    plt.ylabel('F1-Score')
    plt.title('Comparação de F1-Score por Classe e Modelo')
    plt.xticks(x, class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('f1_by_class_comparison.png')
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
    
    print("\nAnálise dos resultados concluída e salva em arquivos de imagem.")
    
    return {
        'best_model': best_model,
        'best_accuracy': accuracies[best_model_idx]
    }

# Função principal que executa o pipeline completo
def main():
    """
    Função principal que executa todo o pipeline de classificação de texto
    """
    print("=== Iniciando o pipeline de classificação de texto ===")
    start_time = time.time()
    
    # Caminhos para os conjuntos de dados
    train_path = 'ag_news_csv/train.csv'
    test_path = 'ag_news_csv/test.csv'
    
    # 1. Carregar os dados
    train_df, test_df = load_data(train_path, test_path)
    
    # 2. Pré-processar o texto
    train_df = preprocess_text(train_df)
    test_df = preprocess_text(test_df)
    
    # 3. Criar vetorizações (Bag of Words e TF-IDF)
    X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, y_train, y_test = create_vectorizations(train_df, test_df)
    
    # 4. Treinar e avaliar os modelos
    results = train_and_evaluate_models(X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # 5. Visualizar os resultados
    metrics_summary = visualize_results(results)
    
    # 6. Análise detalhada dos resultados
    analysis_results = detailed_analysis(results, train_df, test_df)
    
    # Salvar resumo das métricas em CSV
    metrics_summary.to_csv('model_metrics_summary.csv', index=False)
    
    total_time = time.time() - start_time
    print(f"\n=== Pipeline de classificação concluído em {total_time/60:.2f} minutos ===")
    print(f"O melhor modelo foi {analysis_results['best_model']} com acurácia de {analysis_results['best_accuracy']:.4f}")
    print("Todas as visualizações e métricas foram salvas em arquivos para análise posterior.")

if __name__ == "__main__":
    main()
# Text Classifier

Este é um projeto de classificação de texto desenvolvido como parte da Tarefa 7 de TIA.

## Requisitos

- Python 3.x
- UV (gerenciador de pacotes Python)

## Configuração do Ambiente

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd text-classifier
```

2. Inicialize o projeto com UV:
```bash
uv init
```

3. Instale as dependências:
```bash
uv pip install -r requirements.txt
```

## Desenvolvimento

Para adicionar novas dependências:
```bash
uv pip install nome_do_pacote
```

Para atualizar o arquivo requirements.txt:
```bash
uv pip freeze > requirements.txt
```

## Executando o Projeto

Para executar o projeto:
```bash
python main.py
```

## Estrutura do Projeto

- `main.py`: Arquivo principal do projeto
- `requirements.txt`: Lista de dependências do projeto
- `pyproject.toml`: Configurações do projeto Python

## Contribuindo

1. Crie uma nova branch para sua feature:
```bash
git checkout -b feature/nome-da-feature
```

2. Faça commit das suas alterações:
```bash
git add .
git commit -m "Descrição da alteração"
```

3. Faça push para a branch:
```bash
git push origin feature/nome-da-feature
```

4. Abra um Pull Request

## Comandos Úteis do UV

- `uv pip install`: Instala pacotes
- `uv pip uninstall`: Remove pacotes
- `uv pip list`: Lista pacotes instalados
- `uv pip show`: Mostra informações detalhadas de um pacote
- `uv pip freeze`: Lista todas as dependências instaladas
- `uv pip check`: Verifica conflitos de dependências

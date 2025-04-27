# Text Classifier

Este é um projeto de classificação de texto desenvolvido como parte da Tarefa 7 de TIA.

## Configuração Rápida

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd text-classifier
```

2. Crie e ative o ambiente virtual:
```bash
uv venv
.venv\Scripts\activate  # No Windows
source .venv/bin/activate  # No Linux/Mac
```

3. Instale as dependências exatas do arquivo de lock:
```bash
uv pip install -r uv.lock
```

Pronto! Seu ambiente está configurado e pronto para uso.

## Desenvolvimento

### Adicionando Novas Dependências

1. Instale o novo pacote:
```bash
uv pip install nome_do_pacote
```

2. Atualize os arquivos de dependências:
```bash
uv pip freeze > requirements.txt
uv pip compile requirements.txt -o uv.lock
```

### Comandos Úteis

- Verificar pacotes instalados:
```bash
uv pip list
```

- Verificar conflitos de dependências:
```bash
uv pip check
```

- Desativar o ambiente virtual:
```bash
deactivate
```

## Estrutura do Projeto

- `main.py`: Arquivo principal do projeto
- `requirements.txt`: Lista de dependências diretas do projeto
- `uv.lock`: Arquivo de lock com versões exatas de todas as dependências (diretas e indiretas)
- `pyproject.toml`: Configurações do projeto Python

## Sobre o uv.lock

O arquivo `uv.lock` é um arquivo de lock do UV que garante que todas as pessoas que clonam o projeto tenham exatamente as mesmas versões de todas as dependências. Ele:

- Inclui todas as dependências diretas e indiretas
- Armazena versões exatas e hashes de verificação
- Garante reproduzibilidade total do ambiente
- Acelera o processo de instalação das dependências

## Contribuindo

1. Crie uma nova branch:
```bash
git checkout -b feature/nome-da-feature
```

2. Faça commit das alterações:
```bash
git add .
git commit -m "Descrição da alteração"
```

3. Faça push e abra um Pull Request:
```bash
git push origin feature/nome-da-feature
```

## Comandos Úteis do UV

- `uv pip install`: Instala pacotes
- `uv pip uninstall`: Remove pacotes
- `uv pip list`: Lista pacotes instalados
- `uv pip show`: Mostra informações detalhadas de um pacote
- `uv pip freeze`: Lista todas as dependências instaladas
- `uv pip check`: Verifica conflitos de dependências

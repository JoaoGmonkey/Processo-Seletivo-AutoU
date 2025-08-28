# Classificador de Emails com IA (Flask)

Aplicação web simples que **classifica** emails como **Produtivo** ou **Improdutivo** e **sugere respostas automáticas**.

##  Funcionalidades
- Upload de **.txt** ou **.pdf** ou colar texto diretamente.
- **Pré-processamento NLP** (remoção de stopwords, stemming leve).
- **Classificação** via:
  - 🔸 **Regra básica** (fallback – funciona offline).
  - 🔹 **Hugging Face Zero-Shot** (opcional – `USE_HF=1`).
  - 🔹 **OpenAI** (opcional – `USE_OPENAI=1`).
- **Sugestão de resposta** adequada à categoria.
- Interface moderna e responsiva.

##  Como rodar
### 1) Clonar/baixar o repositório
```bash
git clone <seu-fork-ou-repo>.git
cd email-ai-app
```

> Alternativamente, extraia o `.zip` deste projeto e entre na pasta.

### 2) Criar ambiente e instalar dependências
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

você pode comentar `transformers`, `torch` e `openai` no `requirements.txt` e usar apenas a regra básica.

### 3) (Opcional) Ativar Hugging Face ou OpenAI
#### Hugging Face (zero-shot)
```bash
export USE_HF=1  # Windows PowerShell: $env:USE_HF="1"
# Primeiro run pode baixar o modelo "facebook/bart-large-mnli"
```

#### OpenAI
```bash
export USE_OPENAI=1
export OPENAI_API_KEY="sua_chave"
```

> **Importante:** Não ative `USE_HF` e `USE_OPENAI` ao mesmo tempo

### 4) Executar
```bash
python app/app.py
```
Abra http://localhost:5000 no navegador.

## 📂 Estrutura
```
email-ai-app/
├─ app/
│  ├─ app.py          # Flask e rotas
│  ├─ ai.py           # Classificação e sugestão de resposta
│  └─ nlp.py          # Leitura de arquivos e pré-processamento
├─ app/templates/
│  └─ index.html      # Interface
├─ app/static/
│  └─ styles.css      # Estilos
├─ sample_emails/     # Exemplos
├─ requirements.txt
└─ README.md
```

##  Dados de exemplo
A pasta `sample_emails/` contém exemplos:
- `produtivo_suporte.txt`: solicitação de suporte (Produtivo).
- `improdutivo_agradecimento.txt`: agradecimento (Improdutivo).

##  Observações
- O classificador por regra é apenas um **baseline**. Para mais qualidade, use **Hugging Face** (zero-shot) ou **OpenAI**.
- PDFs são lidos via `pdfminer.six`. Se um PDF malformado falhar, o texto pode vir vazio.

##  Segurança
- Limite de upload de 10 MB.
- Extensões permitidas: `.txt`, `.pdf`.



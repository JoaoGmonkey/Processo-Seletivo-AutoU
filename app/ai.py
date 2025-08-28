import os
from typing import Tuple, Optional

# Optional: HuggingFace zero-shot pipeline or OpenAI API. Fallback: rule-based.
USE_HF = os.getenv("USE_HF", "0") == "1"
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"

LABELS = ["Produtivo", "Improdutivo"]

def _rule_based_classifier(text: str, original_text: str) -> Tuple[str, float, str]:
    """
    Very small heuristic as a baseline.
    """
    productive_keywords = [
        "erro", "bug", "suporte", "ajuda", "pendente", "urgente", "responder", "prazo",
        "solicitação", "solicitacao", "chamado", "ticket", "atualização", "atualizacao",
        "acesso", "senha", "login", "implementação", "implementacao", "dúvida", "duvida",
        "orçamento", "orcamento", "cotação", "cotacao", "agendar", "reunião", "reuniao",
        "fatura", "pagamento", "bloqueio"
    ]
    unproductive_keywords = [
        "obrigado", "agradeço", "agradeco", "parabéns", "parabens", "feliz aniversário",
        "feliz aniversario", "boas festas", "bom dia", "boa tarde", "boa noite", "atenciosamente"
    ]

    original_lower = original_text.lower()

    prod_hits = sum(k in original_lower for k in productive_keywords)
    impr_hits = sum(k in original_lower for k in unproductive_keywords)

    if prod_hits > impr_hits:
        return "Produtivo", min(0.99, 0.6 + 0.1 * prod_hits), f"Palavras indicativas de ação detectadas ({prod_hits})."
    if impr_hits > prod_hits:
        return "Improdutivo", min(0.99, 0.6 + 0.1 * impr_hits), f"Mensagem parece de cordialidade/sem ação ({impr_hits})."
    # Fallback by length and presence of question mark
    if "?" in original_lower or len(original_lower) > 120:
        return "Produtivo", 0.55, "Pergunta ou conteúdo mais extenso sugere necessidade de ação."
    return "Improdutivo", 0.55, "Sem indícios claros de ação; mensagem breve e cordial."

def _hf_zero_shot(text: str) -> Tuple[str, float, str]:
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = clf(text, LABELS, multi_label=False)
    # result has 'labels' and 'scores'
    label = result["labels"][0]
    score = float(result["scores"][0])
    return label, score, "Classificação via zero-shot (BART-MNLI)."

def _openai_classifier(text: str) -> Tuple[str, float, str]:
    # Minimal example using OpenAI responses API (user must set OPENAI_API_KEY and install openai>=1.0)
    from openai import OpenAI
    client = OpenAI()
    prompt = f"Classifique o email como 'Produtivo' ou 'Improdutivo'. Apenas responda com uma das duas palavras.\n\nEmail:\n{text}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    label = resp.choices[0].message.content.strip()
    if label not in LABELS:
        label = "Produtivo" if "?" in text or len(text) > 120 else "Improdutivo"
    return label, 0.7, "Classificação via OpenAI Chat Completions."

def classify_email(processed_text: str, original_text: str) -> Tuple[str, Optional[float], str]:
    try:
        if USE_OPENAI:
            return _openai_classifier(original_text)
        if USE_HF:
            return _hf_zero_shot(original_text)
    except Exception as e:
        # Fall back to rule-based if external model fails
        pass
    return _rule_based_classifier(processed_text, original_text)

def suggest_reply(category: str, original_text: str) -> str:
    if category == "Produtivo":
        # A structured template using extracted intent keywords
        return (
            "Olá! Obrigado pela mensagem.\n\n"
            "Entendi sua solicitação. Para agilizar, poderia confirmar:\n"
            "• Descrição do problema/pedido:\n"
            "• Sistema/versão envolvido(a):\n"
            "• Evidências (prints/logs), se houver:\n\n"
            "Assim que recebermos essas informações, daremos sequência e retornaremos com a solução ou próximos passos.\n"
            "Atenciosamente,"
        )
    else:
        return (
            "Olá! Muito obrigado pela mensagem.\n\n"
            "Agradecemos o contato. Ficamos felizes com o retorno e permanecemos à disposição caso precise de algo.\n"
            "Abraços,"
        )

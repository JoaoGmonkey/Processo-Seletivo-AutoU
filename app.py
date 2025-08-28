

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from app.nlp import extract_text_from_upload, preprocess_text
from app.ai import classify_email, suggest_reply

ALLOWED_EXTENSIONS = {"txt", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/process", methods=["POST"])
    def process():
        # Get text input or file
        text_input = request.form.get("email_text", "").strip()
        content = ""
        if "email_file" in request.files and request.files["email_file"]:
            f = request.files["email_file"]
            if f.filename and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                content = extract_text_from_upload(f.stream, filename)
            elif f.filename:
                return jsonify({"error": "Formato de arquivo não suportado. Envie .txt ou .pdf."}), 400

        if not content and text_input:
            content = text_input

        if not content:
            return jsonify({"error": "Nenhum conteúdo de email fornecido."}), 400

        # Preprocess & classify
        processed = preprocess_text(content)
        label, score, rationale = classify_email(processed, original_text=content)
        reply = suggest_reply(label, original_text=content)

        return jsonify({
            "category": label,
            "confidence": round(float(score), 3) if score is not None else None,
            "rationale": rationale,
            "suggested_reply": reply,
            "original_excerpt": (content[:500] + ("..." if len(content) > 500 else ""))
        })

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

from app.lib.cache import cache, cache_key_prefix
from app.lib.detect_llm_usage import detect_llm_usage
from app.main import bp
from flask import render_template, request


@bp.route("/")
@cache.cached(key_prefix=cache_key_prefix)
def index():
    return render_template("main/index.html")


@bp.route("/detect-llm/", methods=["GET", "POST"])
def detect_llm():
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        probability = detect_llm_usage(input_text)
        probability_percentage = round(probability * 100)
        return render_template(
            "main/llm.html", input_text=input_text, score=probability_percentage
        )
    return render_template("main/llm.html")

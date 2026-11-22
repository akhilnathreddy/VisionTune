"""VisionTune AI demo: Streamlit frontend for the FastAPI backend.

    BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 300
LABELS = {"base": "Base Qwen2.5-VL-3B", "qlora": "QLoRA (ScienceQA)", "lora": "LoRA (ScienceQA)"}

st.set_page_config(page_title="VisionTune AI", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{BACKEND}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        return None, f"Backend unreachable at {BACKEND} ({type(exc).__name__})."
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        return None, f"{r.status_code}: {detail}"
    return r.json(), None


def ask_form(key: str):
    upload = st.file_uploader("Scientific image or diagram", type=["png", "jpg", "jpeg", "webp"], key=f"{key}_img")
    if upload:
        st.image(upload, width=420)
    question = st.text_area("Question", key=f"{key}_q", max_chars=1000)
    context = st.text_input("Context (optional)", key=f"{key}_ctx")
    choices = st.text_area("Answer options, one per line (optional; the models were trained on multiple choice)",
                           key=f"{key}_choices", height=110)
    opts = [c.strip() for c in choices.split("\n") if c.strip()]
    return upload, question, context, opts


def payload(upload, question, context, opts):
    files = {"image": (upload.name, upload.getvalue(), upload.type)}
    data = {"question": question, "context": context or "", "choices": json.dumps(opts) if opts else ""}
    return files, data


def show_prediction(p: dict, idx: int):
    st.markdown(f"**{LABELS.get(p['model'], p['model'])}**")
    st.markdown(f"**Answer:** {p.get('answer') or '_could not parse an answer_'}")
    if p.get("explanation"):
        st.markdown(f"**Explanation:** {p['explanation']}")
    st.caption(f"{p['latency_s']:.2f} s · {p['new_tokens']} tokens · parsed: {p['extraction']}")
    with st.expander("Raw model output"):
        st.code(p["raw_output"])
    cols = st.columns(3)
    if cols[0].button("👍 Correct", key=f"up_{idx}_{p['model']}"):
        api("POST", "/feedback", json={"model": p["model"], "question": st.session_state.get("last_q", ""),
                                       "answer": p.get("answer"), "correct": True})
        st.toast("Thanks for the feedback")
    if cols[1].button("👎 Wrong", key=f"down_{idx}_{p['model']}"):
        api("POST", "/feedback", json={"model": p["model"], "question": st.session_state.get("last_q", ""),
                                       "answer": p.get("answer"), "correct": False})
        st.toast("Thanks, recorded as incorrect")
    cols[2].download_button("Download JSON", json.dumps(p, indent=2), file_name=f"visiontune_{p['model']}.json",
                            mime="application/json", key=f"dl_{idx}_{p['model']}")


page = st.sidebar.radio("Page", ["Playground", "Compare models", "Evaluation", "About"])
st.sidebar.warning("Experimental research demo. Answers are not verified scientific facts.")
health, err = api("GET", "/health")
st.sidebar.caption(f"Backend: {BACKEND} · {'🟢 ' + health['status'] if health else '🔴 offline'}")

if page == "Playground":
    st.title("VisionTune AI: scientific image Q&A")
    models, err = api("GET", "/models")
    variants = models["variants"] if models else ["qlora", "lora", "base"]
    default = models["default"] if models else "qlora"
    upload, question, context, opts = ask_form("play")
    model = st.selectbox("Model", variants, index=variants.index(default) if default in variants else 0,
                         format_func=lambda v: LABELS.get(v, v))
    if st.button("Generate answer", type="primary", disabled=not (upload and question.strip())):
        files, data = payload(upload, question, context, opts)
        with st.spinner("Thinking…"):
            pred, err = api("POST", "/predict", files=files, data={**data, "model": model})
        if err:
            st.error(err)
        else:
            st.session_state["last_q"] = question
            st.session_state.setdefault("history", []).insert(0, {"question": question, **pred})
    for i, h in enumerate(st.session_state.get("history", [])[:5]):
        with st.container(border=True):
            st.caption(f"Q: {h['question']}")
            show_prediction(h, i)

elif page == "Compare models":
    st.title("Side-by-side comparison")
    upload, question, context, opts = ask_form("cmp")
    if st.button("Compare all models", type="primary", disabled=not (upload and question.strip())):
        files, data = payload(upload, question, context, opts)
        with st.spinner("Running every model…"):
            res, err = api("POST", "/compare", files=files, data=data)
        if err:
            st.error(err)
        else:
            st.session_state["last_q"] = question
            for col, p in zip(st.columns(len(res["predictions"])), res["predictions"]):
                with col:
                    show_prediction(p, 1000)

elif page == "Evaluation":
    st.title("Measured results")
    metrics, err = api("GET", "/metrics")
    if err:
        st.info("Backend offline; showing the committed result files.")
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from visiontune.serving.results import summary
        metrics = summary()
    plot = ROOT / "results/plots/scaling_curve.png"
    if plot.exists():
        st.image(str(plot), caption="ScienceQA validation accuracy vs training data")
    for key, title in (("scienceqa_test", "ScienceQA test (in-domain, n = 2,017)"),
                       ("ai2d_test", "AI2D test (cross-dataset, n = 3,088)")):
        if key in metrics:
            st.subheader(title)
            st.table([{"model": LABELS.get(k, k), "accuracy %": f"{100 * v['accuracy']:.1f}",
                       "95% CI": f"{100 * v['ci95_low']:.1f}–{100 * v['ci95_high']:.1f}",
                       "strict format %": f"{100 * (v.get('strict_format') or 0):.1f}"}
                      for k, v in metrics[key]["models"].items()])
    if "scienceqa_validation_scaling" in metrics:
        st.subheader("Training runs (ScienceQA validation, n = 2,097)")
        st.table([{"method": r["method"], "train examples": f"{r['n_train']:,}", "accuracy %": f"{100 * r['accuracy']:.1f}",
                   "train min": f"{r['train_minutes']:.0f}", "peak train VRAM GB": f"{r['peak_train_vram_gb']:.2f}"}
                  for r in metrics["scienceqa_validation_scaling"]])

else:
    st.title("About VisionTune AI")
    st.markdown("""
Parameter-efficient fine-tuning of **Qwen2.5-VL-3B-Instruct** for scientific visual question answering.

- **Training:** 6,218 ScienceQA image questions; LoRA (fp16 base) and QLoRA (4-bit NF4 base), r = 16 on the
  language model's attention projections, 2 epochs on one Kaggle T4.
- **ScienceQA test:** base 81.1% → QLoRA 91.5% / LoRA 90.8%.
- **AI2D (unseen benchmark):** no gain (base 79.3%, QLoRA 78.5%, LoRA 77.9%). The adapters are ScienceQA specialists.
- **Limitations:** trained on multiple-choice questions only, so free-form answers are outside the evaluated setting.
  Explanations imitate ScienceQA's style and can sound confident when wrong. Research and non-commercial use only
  (Qwen Research License, ScienceQA CC BY-NC-SA).

Code and full reports: the project's GitHub repository (`results/reports/`).
""")

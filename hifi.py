"""
SmartHealth AI - Clinical Decision Support & Multilingual Frontline Triage
Rewritten to follow the Architecture & UX Specification and lo-fi wireframes.

Pipeline (per spec):
  1. Dialect processing   -> optional ASR + LLM dialect detection / translation
  2. Telemetry ingestion  -> deterministic vitals alerts (hypoxia, fever, ...)
  3. Knowledge grounding  -> retrieval over the guideline PDF (if provided)
  4. Formatted output     -> Dialect -> Statement -> Translation -> Vitals
                             -> Conclusion -> Source

Setup (Colab or local):
    pip install "gradio>=5,<6" openai pypdf
    export LLM_API_KEY="..."        # never hard-code keys
    python smarthealth_app.py
Optional:
    GUIDELINE_PDF=Nigerian-Standard-medical-guideline.pdf   (enables RAG + citation)
    ASR_MODEL=whisper-1                                     (enables mic transcription)
"""

import json
import os
import re

import gradio as gr
import openai

# ------------------------------------------------------------------ CONFIG --
BASE_URL = os.getenv("LLM_BASE_URL", "https://litellm.codeforafrica.org")
MODEL = os.getenv("LLM_MODEL", "bedrock-qwen-qwen3-235b-a22b-2507-v1-0")
ASR_MODEL = os.getenv("ASR_MODEL", "")  # empty = voice transcription disabled
GUIDELINE_PDF = os.getenv("GUIDELINE_PDF", "Nigerian-Standard-medical-guideline.pdf")


def _get_api_key() -> str:
    key = "sk-wb1GJSNkuaA5eNA5ewu-Rg"
    if not key:
        try:  # Google Colab secrets
            from google.colab import userdata  # type: ignore

            key = userdata.get("LLM_API_KEY")
        except Exception:
            pass
    return key or ""


GRADIO_MAJOR = int(gr.__version__.split(".")[0])


def compat(cls, *args, **kwargs):
    """Build a Gradio component, dropping any keyword this Gradio version rejects."""
    while True:
        try:
            return cls(*args, **kwargs)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m or m.group(1) not in kwargs:
                raise
            kwargs.pop(m.group(1))


client = openai.OpenAI(api_key=_get_api_key() or "missing", base_url=BASE_URL)

MODE_1 = "Mode 1 · Chat"
MODE_2 = "Mode 2 · Chat + Vitals"

# ------------------------------------------------- STAGE 3: KNOWLEDGE (RAG) --
_CHUNKS: list[str] = []


def _load_guidelines() -> None:
    """Split the guideline PDF into ~900 char chunks for simple keyword retrieval."""
    if not os.path.exists(GUIDELINE_PDF):
        return
    try:
        from pypdf import PdfReader

        text = "\n".join((p.extract_text() or "") for p in PdfReader(GUIDELINE_PDF).pages)
        text = re.sub(r"[ \t]+", " ", text)
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for p in paras:
            if len(buf) + len(p) > 900 and buf:
                _CHUNKS.append(buf)
                buf = ""
            buf += p + "\n"
        if buf:
            _CHUNKS.append(buf)
    except Exception as e:  # keep the app usable without the PDF
        print(f"[guidelines] could not load {GUIDELINE_PDF}: {e}")


_load_guidelines()


def retrieve_guidelines(query: str, k: int = 3) -> list[str]:
    words = {w for w in re.findall(r"[a-z]{4,}", query.lower())}
    if not words or not _CHUNKS:
        return []
    scored = sorted(
        ((sum(c.lower().count(w) for w in words), c) for c in _CHUNKS),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c for s, c in scored[:k] if s > 0]


# ------------------------------------------------ STAGE 2: TELEMETRY ALERTS --
def vitals_alerts(age, sys_bp, dia_bp, spo2, temp, hr) -> list[str]:
    """Rule-based flags, computed in code so they never depend on the LLM."""
    a = []
    try:
        if spo2 is not None and spo2 < 90:
            a.append(f"SpO₂ {spo2:g}% – hypoxia")
        elif spo2 is not None and spo2 < 94:
            a.append(f"SpO₂ {spo2:g}% – low")
        if temp is not None and temp >= 38:
            a.append(f"Temp {temp:g}°C – fever")
        elif temp is not None and temp < 35.5:
            a.append(f"Temp {temp:g}°C – hypothermia")
        if sys_bp is not None and dia_bp is not None:
            if sys_bp < 90 or dia_bp < 60:
                a.append(f"BP {sys_bp:g}/{dia_bp:g} – hypotension")
            elif sys_bp >= 180 or dia_bp >= 120:
                a.append(f"BP {sys_bp:g}/{dia_bp:g} – severe hypertension")
            elif sys_bp >= 140 or dia_bp >= 90:
                a.append(f"BP {sys_bp:g}/{dia_bp:g} – hypertension")
        if hr is not None:
            hi, lo = (160, 80) if (age is not None and age < 1) else (100, 60)
            if hr > hi:
                a.append(f"HR {hr:g} bpm – tachycardia")
            elif hr < lo:
                a.append(f"HR {hr:g} bpm – bradycardia")
    except TypeError:
        pass
    return a


# -------------------------------------------------------------- STAGE 1: ASR --
def transcribe(audio_path: str | None) -> str:
    if not audio_path or not ASR_MODEL:
        return ""
    try:
        with open(audio_path, "rb") as f:
            r = client.audio.transcriptions.create(model=ASR_MODEL, file=f)
        return (r.text or "").strip()
    except Exception as e:
        print(f"[asr] {e}")
        return ""


# --------------------------------------------------------------- LLM + CARD --
SYSTEM_PROMPT = """You are SmartHealth AI, a clinical decision-support assistant for frontline
health workers in Nigeria. The input may be Hausa, Yoruba, Igbo, Nigerian Pidgin or English.

Reply with ONE JSON object and nothing else (no markdown fences), with keys:
  "dialect":      detected language/dialect name
  "statement":    the original text, unchanged
  "translation":  English translation (same as statement if already English)
  "vitals_summary": 1-2 plain sentences interpreting the vitals, or "" if none were given
  "conclusion":   short, actionable triage advice: likely concerns, immediate actions,
                  danger signs, and whether to refer urgently. Simple wording.
  "urgency":      one of "emergency", "urgent", "routine"

Base the conclusion on the GUIDELINE EXCERPTS when provided. If none are provided, give
cautious general guidance and say that it is not guideline-verified. Never invent doses.
"""

URGENCY = {
    "emergency": "🔴 **Emergency – refer / act now**",
    "urgent": "🟠 **Urgent – see a clinician today**",
    "routine": "🟢 **Routine**",
}


def call_llm(user_text, vitals_block, excerpts):
    parts = []
    if vitals_block:
        parts.append("PATIENT VITALS:\n" + vitals_block)
    if excerpts:
        parts.append("GUIDELINE EXCERPTS:\n" + "\n---\n".join(excerpts))
    parts.append("PATIENT / WORKER STATEMENT:\n" + user_text)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        max_tokens=700,
        temperature=0.2,
    )
    raw = r.choices[0].message.content or ""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"conclusion": raw}


def render_card(d, user_text, vitals_line, alerts, grounded):
    L = []
    L.append(f"🎤 **Input:** “{user_text}”")
    L.append("")
    L.append(f"🗣️ **Detected dialect:** {d.get('dialect', '—')}")
    L.append(f"📝 **Statement:** {d.get('statement', user_text)}")
    L.append(f"🌍 **Translation:** {d.get('translation', '—')}")
    L.append("")
    L.append("📊 **Vitals**")
    if vitals_line:
        L.append(vitals_line)
        if alerts:
            L.append("")
            L.append("⚠️ " + " · ".join(alerts))
        if d.get("vitals_summary"):
            L.append("")
            L.append(d["vitals_summary"])
    else:
        L.append("No vitals provided.")
    L.append("")
    L.append("🩺 **AI conclusion / triage**")
    L.append(URGENCY.get(d.get("urgency", ""), ""))
    L.append("")
    L.append(d.get("conclusion", "No conclusion returned."))
    L.append("")
    L.append("📚 **Reference**")
    if grounded:
        L.append(f"`{os.path.basename(GUIDELINE_PDF)}`")
    else:
        L.append("_Guideline document not loaded – advice is general and not guideline-verified._")
    L.append("")
    L.append("<sub>Decision support only. Confirm with national protocols and clinical judgement.</sub>")
    return "\n".join(L)


# -------------------------------------------------------------------- CHAT --
def respond(message, audio, history, mode, age, weight, sys_bp, dia_bp, spo2, temp, hr, meds):
    history = history or []
    text = (message or "").strip()
    if not text and audio:
        text = transcribe(audio)
        if not text and not ASR_MODEL:
            history.append({"role": "assistant", "content": "⚠️ Voice transcription is not configured. Please type the statement."})
            return history, "", None
    if not text:
        return history, "", None

    use_vitals = mode == MODE_2
    vitals_line, alerts, vitals_block = "", [], ""
    if use_vitals:
        vitals_line = (
            f"Age {age:g} · {weight:g} kg · BP {sys_bp:g}/{dia_bp:g} · SpO₂ {spo2:g}% · "
            f"Temp {temp:g}°C · HR {hr:g} · Meds: {meds or 'none'}"
            if None not in (age, weight, sys_bp, dia_bp, spo2, temp, hr)
            else ""
        )
        alerts = vitals_alerts(age, sys_bp, dia_bp, spo2, temp, hr)
        if vitals_line:
            vitals_block = vitals_line + ("\nAlerts: " + "; ".join(alerts) if alerts else "")

    excerpts = retrieve_guidelines(text + " " + " ".join(alerts))
    history.append({"role": "user", "content": ("🎤 " if audio and not message else "") + text})

    try:
        data = call_llm(text, vitals_block, excerpts)
        card = render_card(data, text, vitals_line, alerts, grounded=bool(excerpts))
    except Exception as e:
        card = f"⚠️ **Could not reach the AI service.** Check your connection and API key, then try again.\n\n`{e}`"
    history.append({"role": "assistant", "content": card})
    return history, "", None


def toggle_vitals(mode):
    return gr.update(visible=(mode == MODE_2), open=(mode == MODE_2))


# ---------------------------------------------------------------------- UI --
CSS = """
:root {
  --sh-green: #0f766e;
  --sh-green-dark: #115e59;
  --sh-bg: #f3f7f6;
  --sh-ink: #0f172a;
}
html, body { background: var(--sh-bg); }
.gradio-container {
  max-width: 760px !important;          /* phone -> tablet -> desktop */
  width: 100% !important;
  margin: 0 auto !important;
  padding: 0 clamp(8px, 3vw, 20px) !important;
  padding-bottom: max(12px, env(safe-area-inset-bottom)) !important;
  padding-top: max(0px, env(safe-area-inset-top)) !important;
}
#sh-header { padding: 14px 4px 4px; }
#sh-header h1 { margin: 0; font-size: clamp(1.25rem, 4.5vw, 1.7rem); color: var(--sh-green-dark); }
#sh-header p  { margin: 2px 0 0; font-size: clamp(.8rem, 2.8vw, .95rem); color: #475569; }

/* Chat area scales with the viewport */
#sh-chat { height: clamp(300px, 52dvh, 620px) !important; }
#sh-chat .message { font-size: clamp(.9rem, 3.2vw, 1rem); line-height: 1.5; }

/* Vitals tray: fluid grid, 2 cols on phones, 4 on wider screens */
#sh-vitals .vitals-grid { display: grid !important; grid-template-columns: repeat(2, 1fr); gap: 8px; }
@media (min-width: 560px) { #sh-vitals .vitals-grid { grid-template-columns: repeat(4, 1fr); } }
#sh-vitals .vitals-grid > * { min-width: 0 !important; }
#sh-vitals .meds { grid-column: 1 / -1; }

/* Touch-friendly controls; 16px inputs stop iOS zoom-on-focus */
.gradio-container input, .gradio-container textarea { font-size: 16px !important; }
.gradio-container button { min-height: 44px; }
#sh-send { flex: 0 0 auto; min-width: 88px !important; }

/* Pin the input bar on small screens like a messaging app */
@media (max-width: 640px) {
  #sh-input-bar { position: sticky; bottom: 0; background: var(--sh-bg); padding: 6px 0; z-index: 5; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate")

HEAD = '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
STYLE_KW = dict(theme=THEME, css=CSS, head=HEAD)  # Gradio 6 takes these in launch()

blocks_kw = {"title": "SmartHealth AI"}
if GRADIO_MAJOR < 6:
    blocks_kw.update(STYLE_KW)

with gr.Blocks(**blocks_kw) as demo:

    gr.HTML(
        '<div id="sh-header"><h1>🏥 SmartHealth AI</h1>'
        "<p>Clinical decision support and multilingual triage for frontline health workers</p></div>"
    )

    mode = gr.Radio([MODE_1, MODE_2], value=MODE_1, label="Switch mode", container=False)

    chat = compat(
        gr.Chatbot,
        elem_id="sh-chat",
        type="messages",          # needed on Gradio 5, removed in 6 (auto-dropped)
        show_label=False,
        placeholder="**Ask by voice or text.**\nExample: “Kaina yana ciwo” or “My wife dey vomit blood”.",
        render_markdown=True,
        show_copy_button=True,    # auto-dropped if unsupported
    )

    with gr.Group(elem_id="sh-input-bar"):
        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                placeholder="Ask… (patient or worker speech)",
                show_label=False, container=False, lines=1, max_lines=4, scale=1,
            )
            send = gr.Button("Send", variant="primary", elem_id="sh-send")

    if ASR_MODEL:
        audio = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak (Hausa / English / Pidgin)")
    else:
        audio = gr.State(None)  # voice disabled until ASR_MODEL is configured

    with gr.Accordion("📊 Tap to enter vitals", open=False, visible=False, elem_id="sh-vitals") as tray:
        with gr.Row(elem_classes=["vitals-grid"]):
            age = gr.Number(label="Age", value=24, minimum=0, maximum=120)
            weight = gr.Number(label="Wt (kg)", value=60, minimum=0, maximum=300)
            sys_bp = gr.Number(label="Sys BP", value=120, minimum=40, maximum=300)
            dia_bp = gr.Number(label="Dia BP", value=80, minimum=20, maximum=200)
            spo2 = gr.Number(label="SpO₂ (%)", value=98, minimum=50, maximum=100)
            temp = gr.Number(label="Temp (°C)", value=37.0, minimum=30, maximum=45)
            hr = gr.Number(label="HR (bpm)", value=72, minimum=20, maximum=250)
            meds = gr.Textbox(label="Meds", value="None", elem_classes=["meds"])

    clear = gr.Button("Clear conversation", variant="secondary", size="sm")

    inputs = [msg, audio, chat, mode, age, weight, sys_bp, dia_bp, spo2, temp, hr, meds]
    outputs = [chat, msg, audio]

    mode.change(toggle_vitals, mode, tray)
    msg.submit(respond, inputs, outputs)
    send.click(respond, inputs, outputs)
    clear.click(lambda: ([], "", None), None, outputs)

if __name__ == "__main__":
    launch_kw = {"share": os.getenv("GRADIO_SHARE", "1") == "1"}
    if GRADIO_MAJOR >= 6:
        launch_kw.update(STYLE_KW)
    demo.queue().launch(**launch_kw)

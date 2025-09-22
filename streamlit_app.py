"""Streamlit-UI für den Transformer-basierten Buch-Adapter."""
from __future__ import annotations
import hashlib
import queue
import threading
import time
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
import torch # Import für Device-Auswahl

try:  # pragma: no cover - optional dependency für Laufzeitprüfung
    from streamlit.runtime.runtime import Runtime
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover - ältere Streamlit-Versionen
    Runtime = None  # type: ignore[assignment]
    get_script_run_ctx = None  # type: ignore[assignment]


def _rerun() -> None:
    """Kompatibler Re-Run-Aufruf für verschiedene Streamlit-Versionen."""

    if not _session_is_active():
        return

    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun is None:  # Korrektur: '===' durch '==' ersetzt
        return

    try:
        rerun()
    except RuntimeError:
        # Tritt u. a. auf, wenn Streamlit während eines Shutdowns keinen Re-Run mehr zulässt.
        pass
    except Exception as exc:  # pragma: no cover - Schutz vor WebSocket-Abbrüchen
        if exc.__class__.__name__ == "WebSocketClosedError":
            # Frontend ist bereits getrennt – weitere Re-Runs würden erneut scheitern.
            pass
        else:
            raise


def _session_is_active() -> bool:
    """Prüft, ob noch eine aktive Streamlit-Sitzung existiert."""

    if Runtime is None or get_script_run_ctx is None:
        # Alte Versionen ohne Runtime-API – wir behalten das bisherige Verhalten bei.
        return True

    try:
        if not Runtime.exists():
            return False
        ctx = get_script_run_ctx()
        if ctx is None or getattr(ctx, "session_id", None) is None:
            return False
        runtime = Runtime.instance()
        return runtime.is_active_session(ctx.session_id)
    except RuntimeError:
        # Runtime.instance() wirft, wenn sie bereits gestoppt wurde.
        return False
    except Exception:
        # Fallback: lieber Re-Runs zulassen als Live-Updates zu verlieren.
        return True


def _clear_widget(key: str) -> None:
    """Entfernt einen Widget-State-Schlüssel, ohne ihn erneut zu setzen."""

    state = st.session_state
    if key in state:
        try:
            del state[key]
        except KeyError:
            pass

# Importe für das Transformer-Modell
from transformer_core import TransformerCore, Tokenizer
from book_ingestion_vsa_adapter_plus_cli import (
    DEMO_TEXT,
    InstrumentedBookAdapter, # Angepasster Adapter für Transformer
    ascii_plot,
)


TRAINING_SEED = 404
REFRESH_DELAY_SEC = 0.6


def _ensure_state() -> None:
    state = st.session_state
    if "metrics_queue" not in state:
        state.metrics_queue = queue.Queue()
    if "status_queue" not in state:
        state.status_queue = queue.Queue()
    if "metrics_history" not in state:
        state.metrics_history: List[Dict[str, float]] = []
    if "interaction_logs" not in state:
        state.interaction_logs: List[str] = []
    if "training_thread" not in state:
        state.training_thread = None
    if "training_running" not in state:
        state.training_running = False
    if "training_status" not in state:
        state.training_status = "Bereit"
    if "training_error" not in state:
        state.training_error = ""
    if "transformer_core" not in state:
        state.transformer_core = None
    if "tokenizer" not in state:
        state.tokenizer = None
    if "adapter" not in state:
        state.adapter = None
    if "book_text" not in state:
        state.book_text = ""
    if "book_text_area" not in state:
        state.book_text_area = state.book_text
    if "last_loaded_brain_digest" not in state:
        state.last_loaded_brain_digest = None
    if "device" not in state:
        state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_adapter(metric_queue: queue.Queue) -> InstrumentedBookAdapter:
    # Initialisiere den Tokenizer und das Transformer-Modell
    tokenizer = Tokenizer()
    model = TransformerCore(
        vocab_size=tokenizer.vocab_size, # Vokabulargröße vom Tokenizer
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.1
    ).to(st.session_state.device) # Modell auf das Gerät verschieben

    # Standard-Optimierer und Kriterium für den Trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Übergabe aller benötigten Parameter an den InstrumentedBookAdapter
    adapter = InstrumentedBookAdapter(
        transformer_core=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion,
        device=st.session_state.device,
    )
    adapter.clear_metric_callbacks()

    def _emit(snapshot: Dict[str, float]) -> None:
        metric_queue.put(snapshot)

    adapter.add_metric_callback(_emit)
    st.session_state.transformer_core = adapter.transformer_core # Speichern des Modells
    st.session_state.tokenizer = adapter.tokenizer # Speichern des Tokenizers
    st.session_state.adapter = adapter
    return adapter


def _attach_adapter(adapter: InstrumentedBookAdapter, metric_queue: queue.Queue) -> None:
    adapter.clear_metric_callbacks()

    def _emit(snapshot: Dict[str, float]) -> None:
        metric_queue.put(snapshot)

    adapter.add_metric_callback(_emit)
    st.session_state.transformer_core = adapter.transformer_core # Speichern des Modells
    st.session_state.tokenizer = adapter.tokenizer # Speichern des Tokenizers
    st.session_state.adapter = adapter


def _start_training(text: str, epochs: int, batch_size: int, learning_rate: float, append: bool = False) -> None:
    state = st.session_state
    metrics_queue: queue.Queue = queue.Queue()
    status_queue: queue.Queue = queue.Queue()
    state.metrics_queue = metrics_queue
    state.status_queue = status_queue
    if append and isinstance(state.adapter, InstrumentedBookAdapter):
        adapter = state.adapter
        _attach_adapter(adapter, metrics_queue)
    else:
        state.metrics_history = []
        adapter = _create_adapter(metrics_queue)

    def _worker() -> None:
        status_queue.put({"type": "status", "value": "Training läuft..."})
        try:
            # Aufruf der Transformer-spezifischen Trainingsmethode
            adapter.ingest_book(text, reset_state=not append)
            adapter.train_on_ingested_data(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            status_queue.put({"type": "status", "value": "Training abgeschlossen"})
        except Exception as exc:  # pragma: no cover - nur zur Anzeige in der UI
            status_queue.put({"type": "error", "value": str(exc)})
        finally:
            status_queue.put({"type": "done", "value": None})

    worker = threading.Thread(target=_worker, name="book_training", daemon=True)
    state.training_thread = worker
    state.training_running = True
    state.training_status = "Training läuft..."
    state.training_error = ""
    worker.start()


def _drain_queues() -> None:
    state = st.session_state
    metrics = state.metrics_queue
    while not metrics.empty():
        state.metrics_history.append(metrics.get())
    status_queue = state.status_queue
    while not status_queue.empty():
        msg = status_queue.get()
        mtype = msg.get("type")
        if mtype == "status":
            state.training_status = msg.get("value", "")
        elif mtype == "error":
            state.training_error = msg.get("value", "")
        elif mtype == "done":
            state.training_running = False


def _render_header() -> None:
    st.title("Transformer-basierter Buch-Lerner – Streamlit UI")
    st.caption(
        "Lade ein Buch hoch oder nutze den Demo-Text, starte das Training und verfolge live die Metriken."
    )


def _render_book_input() -> None:
    state = st.session_state
    with st.expander("📘 Buchquelle", expanded=True):
        col_upload, col_demo = st.columns([2, 1])
        with col_upload:
            uploaded = st.file_uploader("Buch (.txt) hochladen", type=["txt"])
            if uploaded is not None:
                text = uploaded.read().decode("utf-8", errors="ignore")
                state.book_text = text
                state.book_text_area = text
                st.success("Datei übernommen.")
        with col_demo:
            if st.button("Demo-Text laden"):
                state.book_text = DEMO_TEXT
                state.book_text_area = DEMO_TEXT
                _rerun()
        text_value = st.text_area(
            "Buchinhalt (bearbeitbar)",
            value=state.book_text_area,
            height=220,
            key="book_text_area_widget",
        )
        state.book_text = text_value
        state.book_text_area = text_value


def _render_training_controls() -> None:
    state = st.session_state
    with st.expander("⚙️ Training steuern", expanded=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            epochs = st.number_input("Epochen", min_value=1, max_value=100, value=5, step=1)
        with col2:
            batch_size = st.number_input("Batch-Größe", min_value=1, max_value=128, value=16, step=1)
        with col3:
            learning_rate = st.number_input("Lernrate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f")
        with col4:
            st.write("Status:")
            st.info(state.training_status)
            if state.training_error:
                st.error(f"Fehler: {state.training_error}")
        append_training = st.checkbox(
            "An bestehendes Modell anfügen",
            value=False,
            disabled=state.adapter is None,
            help="Führt ein weiteres Buch mit dem aktuellen Modell zusammen statt neu zu initialisieren.",
        )
        start_disabled = state.training_running
        if st.button("🚀 Training starten", disabled=start_disabled):
            if not state.book_text.strip():
                st.warning("Bitte einen Buchtext angeben.")
            else:
                _start_training(state.book_text, int(epochs), int(batch_size), float(learning_rate), append=append_training)
                _rerun()
        if st.button("🔄 System neu initialisieren", disabled=state.training_running):
            state.book_text_area = state.book_text
            state.metrics_history = []
            state.training_status = "Bereit"
            state.training_error = ""
            state.adapter = None
            state.transformer_core = None
            state.tokenizer = None
            st.success("Systemzustand zurückgesetzt.")


def _render_metrics() -> None:
    state = st.session_state
    adapter: InstrumentedBookAdapter | None = state.adapter
    metrics_history = state.metrics_history
    with st.expander("📈 Live-Metriken", expanded=True):
        if metrics_history:
            df = pd.DataFrame(metrics_history).set_index("step")
            # Metriken für Transformer anpassen (Beispiel: loss, accuracy, perplexity)
            if "loss" in df.columns:
                st.line_chart(df[["loss"]])
            if "accuracy" in df.columns:
                st.line_chart(df[["accuracy"]])
            if "perplexity" in df.columns:
                st.line_chart(df[["perplexity"]])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("Noch keine Metriken – starte das Training.")
        if adapter and adapter.metrics.steps:
            cols = st.columns(4)
            last_idx = -1
            cols[0].metric("Schritt", adapter.metrics.steps[last_idx])
            # Direkter Zugriff auf die Metrikwerte, nicht über 'data' Dictionary
            if adapter.metrics.loss:
                cols[1].metric("Verlust (Loss)", f"{adapter.metrics.loss[last_idx]:.4f}")
            if adapter.metrics.accuracy:
                cols[2].metric("Genauigkeit (Accuracy)", f"{adapter.metrics.accuracy[last_idx]:.4f}")
            if adapter.metrics.perplexity:
                cols[3].metric("Perplexität", f"{adapter.metrics.perplexity[last_idx]:.4f}")
            
            # ASCII-Plots für Transformer-Metriken anpassen
            ascii_cols = st.columns(2)
            if adapter.metrics.loss:
                ascii_cols[0].code(ascii_plot(adapter.metrics.loss, label="Verlust", width=48, height=8))
            if adapter.metrics.accuracy:
                ascii_cols[1].code(ascii_plot(adapter.metrics.accuracy, label="Genauigkeit", width=48, height=8))
            if adapter.metrics.perplexity:
                ascii_cols_extra = st.columns(1)
                ascii_cols_extra[0].code(ascii_plot(adapter.metrics.perplexity, label="Perplexität", width=48, height=8))
            
            csv_text = adapter.metrics.to_csv_text()
            st.download_button(
                "📥 Metriken als CSV herunterladen",
                data=csv_text,
                file_name="training_metrics.csv",
                mime="text/csv",
            )


def _render_persistence() -> None:
    state = st.session_state
    adapter: InstrumentedBookAdapter | None = state.adapter
    with st.expander("💾 Modell speichern & laden", expanded=True):
        st.subheader("Aktuellen Zustand sichern")
        if adapter:
            try:
                model_bytes = adapter.export_brain_state()
            except Exception as exc:  # pragma: no cover - Schutz vor Serialisierungsfehlern
                st.error(f"Export fehlgeschlagen: {exc}")
                model_bytes = None
            if model_bytes:
                st.download_button(
                    "🧠 Modell herunterladen",
                    data=model_bytes,
                    file_name="transformer_model.pth",
                    mime="application/octet-stream",
                )
            st.caption("Enthält den Zustand des Transformer-Modells, des Optimierers, des Tokenizers und Metriken.")
        else:
            st.info("Noch kein trainiertes Modell vorhanden.")

        st.subheader("Gespeichertes Modell laden")
        uploaded = st.file_uploader(
            "Modell-Datei auswählen",
            type=["pth", "pt", "bin"],
            key="model_upload",
        )
        if uploaded is not None:
            data = uploaded.getvalue()
            if not data:
                st.warning("Leere Datei hochgeladen.")
            else:
                digest = hashlib.sha256(data).hexdigest()
                if digest == state.last_loaded_brain_digest:
                    st.info("Dieses Modell ist bereits geladen.")
                    _clear_widget("model_upload")
                    _rerun()
                else:
                    try:
                        metrics_queue: queue.Queue = queue.Queue()
                        state.metrics_queue = metrics_queue
                        status_queue: queue.Queue = queue.Queue()
                        state.status_queue = status_queue
                        # Erstellt einen neuen Adapter mit Standard-Modell/Tokenizer,
                        # dessen Zustand dann überschrieben wird.
                        adapter = _create_adapter(metrics_queue)
                        adapter.import_brain_state(data) # Lädt den Zustand in den neuen Adapter
                        state.transformer_core = adapter.transformer_core # Speichern des Modells
                        state.tokenizer = adapter.tokenizer # Speichern des Tokenizers
                        state.metrics_history = adapter.metrics.as_dicts()
                        state.training_status = "Modell geladen"
                        state.training_error = ""
                        state.training_running = False
                        state.last_loaded_brain_digest = digest
                        st.success("Modell erfolgreich geladen.")
                    except Exception as exc:
                        st.error(f"Laden fehlgeschlagen: {exc}")
                    finally:
                        _clear_widget("model_upload")
                        _rerun()


def _render_interactions() -> None:
    state = st.session_state
    adapter: InstrumentedBookAdapter | None = state.adapter
    disable_actions = state.training_running or not adapter
    with st.expander("🧠 Nach dem Training interagieren", expanded=True):
        if not adapter:
            st.info("Nach Trainingsende stehen hier Interaktionen zur Verfügung.")
            return
        st.caption("Nutze die Funktionen des CLI-Tools jetzt direkt in der Web-Oberfläche.")
        
        # Textgenerierung als neue Interaktion
        with st.form("generate_form", clear_on_submit=False):
            st.subheader("Text generieren")
            prompt = st.text_input("Prompt eingeben", key="gen_prompt")
            length = st.number_input("Länge der Generierung", min_value=10, max_value=500, value=100, step=10)
            temperature = st.slider("Temperatur (Kreativität)", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
            submitted = st.form_submit_button("Generieren", disabled=disable_actions)
            if submitted:
                try:
                    generated_text = adapter.generate_text(prompt, max_length=int(length), temperature=float(temperature))
                    st.text_area("Generierter Text", value=generated_text, height=200)
                except Exception as exc:
                    st.error(f"Fehler bei der Generierung: {exc}")
                else:
                    state.interaction_logs.append(f"generate('{prompt}', len={length}, temp={temperature})")

        if state.interaction_logs:
            st.subheader("Aktionsprotokoll")
            st.write("\n".join(state.interaction_logs[-10:]))


def main() -> None:
    st.set_page_config(page_title="Transformer Buchtrainer", layout="wide")
    _ensure_state()
    _drain_queues()
    _render_header()
    _render_book_input()
    _render_training_controls()
    _render_metrics()
    _render_persistence()
    _render_interactions()
    if st.session_state.training_running:
        time.sleep(REFRESH_DELAY_SEC)
        _rerun()


if __name__ == "__main__":
    main()

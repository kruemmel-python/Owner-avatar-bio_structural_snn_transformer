from __future__ import annotations
# ------------------------------------------------------------
# Buch-Lernen (Transformer LLM) mit:
# - Tokenisierung & Embedding
# - Transformer-Architektur
# - Trainings- und Inferenzlogik
# + NEU: Metrik-Logging, CSV-Export, ASCII-Plots, interaktive CLI
# Nur Standardbibliothek, Python 3.12 (mit PyTorch/Transformers)
# ------------------------------------------------------------

import argparse
import csv
import io
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional

# >>> Angepasste Imports für Transformer-Architektur
import transformer_core as tc # Das neue Transformer-Modul
# import book_ingestion_vsa_adapter_plus as adapter_plus # Nicht mehr direkt verwendet, da Tokenisierung im Transformer-Modul ist
# from spaced_repetition_trainer import Trainer # Wird durch TransformerTrainer ersetzt

# ==============================
# 1) ASCII-Plot & Metrik-Logger
# ==============================

def _min_max(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 1.0)
    mn = min(xs)
    mx = max(xs)
    if mx - mn < 1e-9:
        mx = mn + 1.0
    return mn, mx

def ascii_plot(series: List[float], width: int = 64, height: int = 10, label: str = "") -> str:
    """
    Einfache ASCII-„Heatmap“-Zeitreihe: y=0..height-1, x=0..width-1
    skaliert auf min..max. Keine externen Libs.
    """
    if not series:
        return f"{label}: (keine Daten)\n"
    # verdichte/strecke auf width
    N = len(series)
    if N <= width:
        xs = series + [series[-1]] * (width - N)
    else:
        # Mittelung in Blöcken
        block = N / width
        xs = []
        acc = 0.0
        cnt = 0
        idx_target = block
        j = 0
        while len(xs) < width:
            acc += series[j]
            cnt += 1
            if cnt >= idx_target or j == N-1:
                xs.append(acc/cnt)
                acc = 0.0
                cnt = 0
            j = min(N-1, j+1)

    mn, mx = _min_max(xs)
    # Raster
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x, v in enumerate(xs):
        y = 0 if mx-mn < 1e-9 else int((v - mn) / (mx - mn) * (height - 1))
        y = max(0, min(height - 1, y))
        # zeichne Säule bis y
        for yy in range(y + 1):
            ch = " ▂▃▄▅▆▇█"[min(7, max(0, int(yy / max(1, height-1) * 7)))]
            grid[height - 1 - yy][x] = ch
    # Achseninfo
    lines = [f"{label}  min={mn:.4f}  max={mx:.4f}"]
    for row in grid:
        lines.append("".join(row))
    return "\n".join(lines) + "\n"

@dataclass
class MetricsLogger:
    # Angepasste Metriken für Transformer-Modell
    loss:        List[float] = field(default_factory=list)
    perplexity:  List[float] = field(default_factory=list)
    accuracy:    List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    steps:       List[int]   = field(default_factory=list)

    def log(self, step: int, loss_val: float, ppl: float, acc: float, lr: float) -> None:
        self.steps.append(step)
        self.loss.append(loss_val)
        self.perplexity.append(ppl)
        self.accuracy.append(acc)
        self.learning_rate.append(lr)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step","loss","perplexity","accuracy","learning_rate"])
            for i in range(len(self.steps)):
                w.writerow([
                    self.steps[i],
                    f"{self.loss[i]:.6f}",
                    f"{self.perplexity[i]:.6f}",
                    f"{self.accuracy[i]:.6f}",
                    f"{self.learning_rate[i]:.8f}",
                ])

    def to_csv_text(self) -> str:
        """Liefert identische CSV-Daten als String (für Downloads)."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["step","loss","perplexity","accuracy","learning_rate"])
        for i in range(len(self.steps)):
            w.writerow([
                self.steps[i],
                f"{self.loss[i]:.6f}",
                f"{self.perplexity[i]:.6f}",
                f"{self.accuracy[i]:.6f}",
                f"{self.learning_rate[i]:.8f}",
            ])
        return buf.getvalue()

    def as_dicts(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for i in range(len(self.steps)):
                rows.append(
                    {
                        "step": self.steps[i],
                        "loss": self.loss[i],
                        "perplexity": self.perplexity[i],
                        "accuracy": self.accuracy[i],
                        "learning_rate": self.learning_rate[i],
                    }
                )
        return rows

# ==============================
# 2) Instrumentierter Adapter (wrappt den „Plus“-Adapter und Transformer)
# ==============================

class InstrumentedBookAdapter: # Nicht mehr von adapter_plus.BookAdapter erben
    def __init__(
        self,
        transformer_core: tc.TransformerCore,
        tokenizer: tc.Tokenizer,
        optimizer: Optional[tc.torch.optim.Optimizer] = None,
        criterion: Optional[tc.torch.nn.Module] = None,
        device: Optional[tc.torch.device] = None
    ):
        self.tokenizer = tokenizer
        self.metrics = MetricsLogger()
        self._step_counter = 0
        self._metric_callbacks: List[Callable[[Dict[str, float]], None]] = []
        self.ingested_text_data: List[List[int]] = [] # Tokenisierte Daten

        self.device = device or tc.torch.device("cuda" if tc.torch.cuda.is_available() else "cpu")
        self.transformer_core = transformer_core.to(self.device)

        if optimizer is None:
            optimizer = tc.torch.optim.Adam(self.transformer_core.parameters(), lr=1e-4)
        if criterion is None:
            criterion = tc.torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        # Initialisiere den Trainer mit allen erforderlichen Parametern
        self.trainer = tc.TransformerTrainer(
            model=self.transformer_core,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            tokenizer=tokenizer
        )

    def add_metric_callback(self, cb: Callable[[Dict[str, float]], None]) -> None:
        """Registriere Callback, der bei jedem Log ein Snapshot-Dikt erhält."""
        if cb is None:
            return
        self._metric_callbacks.append(cb)

    def clear_metric_callbacks(self) -> None:
        """Entferne alle registrierten Callbacks (z. B. beim Reset)."""
        self._metric_callbacks.clear()

    def _log_now(self, loss: float = 0.0, ppl: float = 0.0, acc: float = 0.0, lr: float = 0.0):
        self.metrics.log(self._step_counter, loss, ppl, acc, lr)
        if self._metric_callbacks:
            payload = {
                "step": self.metrics.steps[-1],
                "loss": self.metrics.loss[-1],
                "perplexity": self.metrics.perplexity[-1],
                "accuracy": self.metrics.accuracy[-1],
                "learning_rate": self.metrics.learning_rate[-1],
            }
            for cb in list(self._metric_callbacks):
                try:
                    cb(payload)
                except Exception:
                    # externe Konsumenten sollen das Logging nicht blockieren
                    pass

    def ingest_book(
        self,
        text: str,
        chapter_size_sents: int = 40, # Diese Parameter werden für die Tokenisierung verwendet
        para_size_sents: int = 6,
        *,
        reset_state: bool = True,
    ) -> None:
        if reset_state:
            self.metrics = MetricsLogger()
            self._step_counter = 0
            self.ingested_text_data = []
            self.trainer.reset_optimizer() # Auch den Optimierer zurücksetzen

        print("-> Tokenisiere Buch...")
        # Hier wird der Text in eine Liste von Token-IDs umgewandelt.
        # Der `adapter_plus` könnte hier für die Segmentierung in Sätze/Absätze helfen,
        # bevor der Tokenizer des Transformers angewendet wird.
        # Für eine einfache Implementierung: gesamter Text wird tokenisiert.
        token_tensor = self.tokenizer.encode(text, truncation=False, padding='do_not_pad')
        token_ids = token_tensor.squeeze(0).tolist()
        self.ingested_text_data.append(token_ids) # Speichern der tokenisierten Daten

        # Anfangslog (mit 0-Werten, da noch kein Training)
        self._log_now(0.0, 0.0, 0.0, 0.0)
        print(f"Ingestion fertig. Token: {len(token_ids)}")

    def train_on_ingested_data(self, epochs: int = 1, batch_size: int = 4, learning_rate: float = 1e-4):
        """Trainiert das Transformer-Modell auf den ingestierten Daten."""
        if not self.ingested_text_data:
            print("Keine Daten zum Trainieren vorhanden. Bitte zuerst ein Buch ingestieren.")
            return

        print(f"Starte Training für {epochs} Epochen...")
        all_token_ids = [token for sublist in self.ingested_text_data for token in sublist]
        total_tokens = len(all_token_ids)
        if total_tokens < 2:
            print("Zu wenige Tokens für das Training.")
            return

        max_seq_len = self.transformer_core.pos_encoder.pe.size(0)
        seq_len = max(2, min(max_seq_len, total_tokens - 1))
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_chunks: List[List[int]] = []
        target_chunks: List[List[int]] = []
        for start in range(0, total_tokens - 1, seq_len):
            end = min(start + seq_len, total_tokens - 1)
            input_chunk = all_token_ids[start:end]
            target_chunk = all_token_ids[start + 1:end + 1]
            if len(input_chunk) < seq_len:
                input_chunk = input_chunk + [pad_id] * (seq_len - len(input_chunk))
            if len(target_chunk) < seq_len:
                target_chunk = target_chunk + [pad_id] * (seq_len - len(target_chunk))
            input_chunks.append(input_chunk)
            target_chunks.append(target_chunk)

        input_tensor = tc.torch.tensor(input_chunks, dtype=tc.torch.long)
        target_tensor = tc.torch.tensor(target_chunks, dtype=tc.torch.long)
        dataset = tc.torch.utils.data.TensorDataset(input_tensor, target_tensor)
        dataloader = tc.torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setze die Trainingsdaten im Trainer
        self.trainer.set_training_data(dataloader)
        self.trainer.set_learning_rate(learning_rate)

        for epoch in range(epochs):
            print(f"Epoche {epoch+1}/{epochs}")
            for loss, ppl, acc, lr in self.trainer.train_batches():
                self._step_counter += 1
                self._log_now(loss, ppl, acc, lr)
                if self._step_counter % 10 == 0:
                    print(f"  Schritt {self._step_counter}: Loss={loss:.4f}, PPL={ppl:.2f}, Acc={acc:.2f}")
        print("Training abgeschlossen.")

    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """Generiert Text basierend auf einem Prompt."""
        print(f"Generiere Text mit Prompt: '{prompt}'...")
        input_ids = self.tokenizer.encode(prompt, truncation=False, padding='do_not_pad').to(self.device)
        generated_ids = self.transformer_core.generate_text(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
        )
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text

    def export_brain_state(self) -> bytes:
        # Speichern des Transformer-Modells, des Optimierers und des Tokenizers
        payload = {
            "transformer_state_dict": self.transformer_core.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "tokenizer_name": self.tokenizer.name, # Speichere den Namen des Tokenizers
            "metrics": {
                "steps": list(self.metrics.steps),
                "loss": list(self.metrics.loss),
                "perplexity": list(self.metrics.perplexity),
                "accuracy": list(self.metrics.accuracy),
                "learning_rate": list(self.metrics.learning_rate),
            },
            "step_counter": self._step_counter,
            "ingested_text_data": [list(chunk) for chunk in self.ingested_text_data],
        }
        # torch.save kann bei BytesIO mit dem Zip-Serialisierer in seltenen Fällen
        # mit einer "unexpected pos"-Meldung abbrechen. Wir versuchen daher zuerst
        # den Standardweg und fallen bei Problemen auf das Legacy-Format zurück.
        class _NonClosingBytesIO(io.BytesIO):
            def close(self) -> None:  # type: ignore[override]
                # torch.save() schließt den Stream nach dem Schreiben.
                # Für BytesIO würde das spätere getvalue() scheitern, daher ignorieren.
                pass

            def real_close(self) -> None:
                super().close()

        buffer = _NonClosingBytesIO()
        try:
            tc.torch.save(payload, buffer)
        except RuntimeError:
            buffer.real_close()
            buffer = _NonClosingBytesIO()
            tc.torch.save(payload, buffer, _use_new_zipfile_serialization=False)
        data = buffer.getvalue()
        buffer.real_close()
        return data

    def import_brain_state(self, data: bytes) -> None:
        buffer = io.BytesIO(data)
        payload = tc.torch.load(buffer, map_location=self.device)
        if not isinstance(payload, dict):
            raise ValueError("Ungültiges Gehirnformat: Erwartet Wörterbuch.")

        # Laden des Transformer-Modells
        self.transformer_core.load_state_dict(payload.pop("transformer_state_dict", {}))
        
        # Laden des Optimierers
        optimizer_state_dict = payload.pop("optimizer_state_dict", None)
        if optimizer_state_dict:
            self.trainer.load_optimizer_state_dict(optimizer_state_dict)

        # Laden des Tokenizers (ggf. neu initialisieren)
        tokenizer_name = payload.pop("tokenizer_name", None)
        if tokenizer_name:
            # Annahme: Tokenizer kann mit einem Namen neu initialisiert werden
            self.tokenizer = tc.Tokenizer(pretrained_model_name_or_path=tokenizer_name)
            # Aktualisiere den Tokenizer im Trainer, falls dieser ihn auch hält
            self.trainer.tokenizer = self.tokenizer
            if isinstance(self.trainer.criterion, tc.torch.nn.CrossEntropyLoss):
                self.trainer.criterion = tc.torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        self.transformer_core.to(self.device)
        self.trainer.model = self.transformer_core
        self.trainer.device = self.device
        
        metrics_payload = payload.pop("metrics", None)
        if isinstance(metrics_payload, dict):
            metrics = MetricsLogger()
            metrics.steps = list(metrics_payload.get("steps", []))
            metrics.loss = list(metrics_payload.get("loss", []))
            metrics.perplexity = list(metrics_payload.get("perplexity", []))
            metrics.accuracy = list(metrics_payload.get("accuracy", []))
            metrics.learning_rate = list(metrics_payload.get("learning_rate", []))
            self.metrics = metrics
        elif isinstance(metrics_payload, MetricsLogger):
            # Fallback für ältere Speicherstände
            self.metrics = metrics_payload
        else:
            self.metrics = MetricsLogger()
        self._step_counter = int(payload.pop("step_counter", 0))
        ingested_payload = payload.pop("ingested_text_data", [])
        self.ingested_text_data = [list(chunk) for chunk in ingested_payload]

        # Falls importierte Metriken leer sind, aktuellen Zustand loggen, damit UI Werte hat.
        if not self.metrics.steps:
            self._log_now(0.0, 0.0, 0.0, 0.0)

# ==============================
# 3) CLI
# ==============================

HELP_TEXT = """
Befehle:
  train [epochen] [batch_size] [lr] – Transformer-Modell trainieren
  generate <prompt>              – Text generieren
  stats                          – letzte Metriken kurz anzeigen
  plot                           – ASCII-Plots (Loss, Perplexity, Accuracy, Learning Rate)
  savecsv <pfad.csv>             – Metriken als CSV exportieren
  help                           – diese Hilfe
  exit                           – beenden
"""

def print_stats(inst: InstrumentedBookAdapter) -> None:
    m = inst.metrics
    if not m.steps:
        print("Noch keine Metriken.")
        return
    i = -1
    print(f"Schritt: {m.steps[i]}")
    print(f"Loss: {m.loss[i]:.4f}")
    print(f"Perplexity: {m.perplexity[i]:.2f}")
    print(f"Accuracy: {m.accuracy[i]:.2f}")
    print(f"Learning Rate: {m.learning_rate[i]:.8f}")

def print_plots(inst: InstrumentedBookAdapter) -> None:
    m = inst.metrics
    print(ascii_plot(m.loss, label="Loss", width=64, height=8))
    print(ascii_plot(m.perplexity,  label="Perplexity", width=64, height=8))
    print(ascii_plot(m.accuracy, label="Accuracy", width=64, height=8))
    print(ascii_plot(m.learning_rate, label="Learning Rate", width=64, height=8))

def run_cli(inst: InstrumentedBookAdapter) -> None:
    print("\nInteraktive CLI – tippe 'help' für Hilfe, 'exit' zum Beenden.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTschüss.")
            break
        if not line:
            continue
        if line.lower() in {"exit","quit"}:
            print("Tschüss.")
            break
        if line.lower() in {"help","?"}:
            print(HELP_TEXT)
            continue
        if line.startswith("train"):
            parts = line.split()
            epochs = int(parts[1]) if len(parts) >= 2 else 1
            batch_size = int(parts[2]) if len(parts) >= 3 else 4
            learning_rate = float(parts[3]) if len(parts) >= 4 else 1e-4
            try:
                inst.train_on_ingested_data(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
                print(f"Training abgeschlossen. Epochen={epochs}, Batch={batch_size}, LR={learning_rate}.")
            except Exception as e:
                print(f"Fehler beim Training: {e}")
            continue
        if line.startswith("generate "):
            prompt = line[len("generate "):].strip()
            try:
                generated_text = inst.generate_text(prompt)
                print(f"Generierter Text:\n{generated_text}")
            except Exception as e:
                print(f"Fehler beim Generieren: {e}")
            continue
        if line == "stats":
            print_stats(inst)
            continue
        if line == "plot":
            print_plots(inst)
            continue
        if line.startswith("savecsv "):
            path = line[len("savecsv "):].strip()
            try:
                inst.metrics.to_csv(path)
                print(f"Gespeichert: {path}")
            except Exception as e:
                print(f"Fehler beim Speichern: {e}")
            continue
        print("Unbekannter Befehl. 'help' zeigt Hilfe.")

# ==============================
# 4) Main
# ==============================

DEMO_TEXT = """
Kapitel Eins. Der Hund bellt im Garten. Die Katze schläft auf dem Sofa. Der Hund jagt die Katze.
Später bellt der Hund nicht. Die Katze läuft in den Garten. Der Junge steht auf und ruft.
Kapitel Zwei. Der Mond scheint über dem See. Der Hund schläft. Die Katze jagt nicht den Hund.
"""

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Buch-Lernen (Transformer LLM) – CLI & Metriken")
    parser.add_argument("--book", type=str, default="", help="Pfad zu buch.txt (UTF-8)")
    parser.add_argument("--demo", action="store_true", help="Demo-Text verwenden (Default, wenn --book fehlt)")
    parser.add_argument("--chapter-size", type=int, default=40, help="Sätze pro Kapitel (Chunking, für Tokenizer)")
    parser.add_argument("--para-size", type=int, default=6, help="Sätze pro Absatz (Chunking, für Tokenizer)")
    parser.add_argument("--model-path", type=str, default="transformer_model.pkl", help="Pfad zum Speichern/Laden des Modells")
    args = parser.parse_args(argv)

    # Initialisiere Transformer-Modell und Tokenizer
    # Für Demo-Zwecke: Ein sehr einfaches Modell und Tokenizer
    # Die Vokabulargröße wird vom Tokenizer bestimmt
    
    # Zuerst den Tokenizer initialisieren, da er die Vokabulargröße für das Modell liefert
    tokenizer = tc.Tokenizer() # Initialisiert den AutoTokenizer
    vocab_size = tokenizer.vocab_size # Holt die tatsächliche Vokabulargröße
    
    d_model = 128
    n_heads = 4
    n_layers = 2
    
    transformer_core = tc.TransformerCore(vocab_size, d_model, n_heads, n_layers)
    # Initialisiere den Adapter mit allen notwendigen Komponenten
    inst = InstrumentedBookAdapter(
        transformer_core=transformer_core,
        tokenizer=tokenizer,
        # Optimierer, Kriterium und Gerät werden vom Trainer selbst initialisiert,
        # wenn sie hier nicht explizit übergeben werden.
        # Dies ist die Standardfunktionalität des TransformerTrainer.
    )

    if os.path.exists(args.model_path):
        print(f"-> Lade Modellzustand von: {args.model_path}")
        try:
            with open(args.model_path, "rb") as f:
                inst.import_brain_state(f.read())
            print("Modellzustand erfolgreich geladen.")
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}. Starte mit neuem Modell.")
            # Bei Fehler mit neuem Modell fortfahren
            # Tokenizer muss neu initialisiert werden, falls der alte nicht geladen werden konnte
            new_tokenizer = tc.Tokenizer()
            new_transformer_core = tc.TransformerCore(new_tokenizer.vocab_size, d_model, n_heads, n_layers)
            inst = InstrumentedBookAdapter(
                transformer_core=new_transformer_core,
                tokenizer=new_tokenizer,
            )


    if args.book and os.path.exists(args.book):
        with open(args.book, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"-> Lese Buch: {args.book}")
    else:
        print("-> Demo-Text wird verwendet. (Nutze --book pfad/zu/buch.txt für eigenes Buch)")
        text = DEMO_TEXT

    t0 = time.time()
    inst.ingest_book(text, chapter_size_sents=args.chapter_size, para_size_sents=args.para_size)
    dt = time.time() - t0
    print(f"\nIngestion fertig. Dauer: {dt:.2f}s")
    print_stats(inst)
    print("\nKurze Plots (ASCII):")
    print_plots(inst)

    run_cli(inst)

    # Speichern des Modellzustands beim Beenden
    print(f"-> Speichere Modellzustand nach: {args.model_path}")
    try:
        with open(args.model_path, "wb") as f:
            f.write(inst.export_brain_state())
        print("Modellzustand erfolgreich gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern des Modells: {e}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

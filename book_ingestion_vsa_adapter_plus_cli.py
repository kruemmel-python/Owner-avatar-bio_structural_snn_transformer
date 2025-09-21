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
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

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

class InstrumentedBookAdapter:  # Nicht mehr von adapter_plus.BookAdapter erben
    def __init__(
        self,
        transformer_core: Optional[tc.TransformerCore] = None,
        tokenizer: Optional[tc.Tokenizer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        **kwargs: object,
    ) -> None:
        """Instrumentierter Adapter für den Transformer-Trainer.

        Die ursprüngliche CLI rief den Adapter mit den Positionsargumenten
        ``transformer_core`` und ``tokenizer`` auf.  Die Streamlit-Oberfläche
        übergibt hingegen Schlüsselwörter wie ``model`` und ``tokenizer``.
        Um beide Varianten zu unterstützen, akzeptiert der Konstruktor
        optionale Schlüsselwörter und normalisiert sie auf die erwarteten
        Attribute.
        """

        if transformer_core is None:
            transformer_core = kwargs.pop("model", None)
        if tokenizer is None:
            tokenizer = kwargs.get("tokenizer")  # Streamlit übergibt diesen Namen bereits
        if transformer_core is None or tokenizer is None:
            raise ValueError("InstrumentedBookAdapter benötigt ein Transformer-Modell und einen Tokenizer.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if optimizer is None:
            optimizer = torch.optim.Adam(transformer_core.parameters(), lr=1e-4)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        self.transformer_core = transformer_core
        # Alias für Kompatibilität mit der Streamlit-Oberfläche
        self.model = transformer_core
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = MetricsLogger()
        self._step_counter = 0
        self._metric_callbacks: List[Callable[[Dict[str, float]], None]] = []
        self.ingested_text_data: List[List[int]] = []  # Tokenisierte Daten

        # Tokenizer warnen bei sehr langen Sequenzen mit einem Hinweis, dass das
        # Modell maximal ``model_max_length`` Tokens gleichzeitig verarbeiten
        # könne. Für das Training zerschneiden wir die Tokens aber später
        # ohnehin in handliche Blöcke. Damit der Hinweis nicht jedes Mal
        # erscheint (obwohl wir ihn sicher beherrschen), erhöhen wir das Limit
        # auf einen konservativen, aber großzügigen Wert.
        try:
            context_window = self.transformer_core.pos_encoder.pe.size(0)
        except AttributeError:
            context_window = 0

        scaled_window = int(context_window) * 64 if context_window else 0
        target_max_length = max(scaled_window, 1_000_000)

        current_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(current_max_length, int):
            if current_max_length < target_max_length:
                try:
                    self.tokenizer.model_max_length = target_max_length
                except (AttributeError, ValueError):
                    pass
        # Einige Tokenizer lesen den Wert zusätzlich aus init_kwargs aus.
        init_kwargs = getattr(self.tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            existing = init_kwargs.get("model_max_length")
            if not isinstance(existing, int) or existing < target_max_length:
                init_kwargs["model_max_length"] = target_max_length

        # Initialisiere den Trainer mit allen erforderlichen Parametern
        self.trainer = tc.TransformerTrainer(
            model=transformer_core,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            tokenizer=tokenizer,
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
        chapter_size_sents: int = 40,
        para_size_sents: int = 6,
        *,
        reset_state: bool = True,
    ) -> None:
        if reset_state:
            self.metrics = MetricsLogger()
            self._step_counter = 0
            self.ingested_text_data = []
            self.trainer.reset_optimizer()

        print("-> Tokenisiere Buch...")
        token_tensor = self.tokenizer.encode(
            text,
            max_length=None,
            truncation=False,
            padding="do_not_pad",
        )
        token_ids = token_tensor.squeeze(0).tolist()
        self.ingested_text_data.append(token_ids)

        # Anfangslog (mit 0-Werten, da noch kein Training)
        self._log_now(0.0, 0.0, 0.0, self.trainer.optimizer.param_groups[0]["lr"])
        print(f"Ingestion fertig. Token: {len(token_ids)}")

    def _prepare_training_dataset(self) -> Optional[TensorDataset]:
        if not self.ingested_text_data:
            return None

        tokens = [token for sequence in self.ingested_text_data for token in sequence]
        if len(tokens) < 2:
            return None

        max_len = min(len(tokens) - 1, self.transformer_core.pos_encoder.pe.size(0))
        pad_id = self.tokenizer.pad_token_id

        inputs: List[List[int]] = []
        targets: List[List[int]] = []

        stride = max_len
        for start in range(0, len(tokens) - 1, stride):
            chunk = tokens[start : start + max_len + 1]
            if len(chunk) < 2:
                continue
            input_seq = chunk[:-1]
            target_seq = chunk[1:]

            if len(input_seq) < max_len:
                pad_amount = max_len - len(input_seq)
                input_seq += [pad_id] * pad_amount
                target_seq += [pad_id] * pad_amount

            inputs.append(input_seq[:max_len])
            targets.append(target_seq[:max_len])

        if not inputs:
            return None

        input_tensor = torch.tensor(inputs, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        return TensorDataset(input_tensor, target_tensor)

    def train_on_ingested_data(
        self,
        text: Optional[str] = None,
        *,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        reset_optimizer: bool = True,
    ) -> None:
        """Trainiert das Transformer-Modell auf den ingestierten Daten."""

        if text is not None:
            self.ingest_book(text, reset_state=reset_optimizer)

        dataset = self._prepare_training_dataset()
        if dataset is None:
            print("Keine Daten zum Trainieren vorhanden. Bitte zuerst ein Buch ingestieren.")
            return

        if reset_optimizer:
            self.trainer.reset_optimizer(learning_rate)
        else:
            self.trainer.set_learning_rate(learning_rate)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.trainer.set_training_data(data_loader)

        print(f"Starte Training für {epochs} Epochen...")
        for epoch in range(epochs):
            print(f"Epoche {epoch + 1}/{epochs}")
            for batch_idx, (input_ids, target_ids) in enumerate(data_loader, start=1):
                loss = self.trainer.train_step(input_ids, target_ids)

                with torch.no_grad():
                    input_device = input_ids.to(self.device)
                    target_device = target_ids.to(self.device)
                    logits = self.transformer_core(input_device)
                    predictions = logits.argmax(dim=-1)
                    mask = target_device != self.tokenizer.pad_token_id
                    if mask.any():
                        correct = (predictions == target_device) & mask
                        accuracy = correct.float().sum().item() / mask.float().sum().item()
                    else:
                        accuracy = 0.0

                perplexity = math.exp(loss) if loss < 20 else float("inf")
                current_lr = self.trainer.optimizer.param_groups[0]["lr"]

                self._step_counter += 1
                self._log_now(loss, perplexity, accuracy, current_lr)

                if self._step_counter % 10 == 0:
                    print(
                        f"  Schritt {self._step_counter}: Loss={loss:.4f}, PPL={perplexity:.2f}, Acc={accuracy:.2f}, LR={current_lr:.6f}"
                    )

        print("Training abgeschlossen.")

    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """Generiert Text basierend auf einem Prompt."""
        print(f"Generiere Text mit Prompt: '{prompt}'...")
        if not prompt:
            return ""

        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.transformer_core.pos_encoder.pe.size(0),
            truncation=True,
            padding="do_not_pad",
        ).to(self.device)

        generated_ids = self.trainer.generate_text(prompt_ids, max_new_tokens=max_length, temperature=temperature)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    # Kompatibilitätsalias für die Streamlit-App
    def export_model_state(self) -> bytes:
        return self.export_brain_state()

    def import_model_state(self, data: bytes) -> None:
        self.import_brain_state(data)

    def export_brain_state(self) -> bytes:
        # Speichern des Transformer-Modells, des Optimierers und des Tokenizers
        payload = {
            "transformer_state_dict": self.transformer_core.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "tokenizer_name": self.tokenizer.name, # Speichere den Namen des Tokenizers
            "metrics": self.metrics,
            "step_counter": self._step_counter,
            "ingested_text_data": self.ingested_text_data,
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def import_brain_state(self, data: bytes) -> None:
        payload = pickle.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Ungültiges Gehirnformat: Erwartet Wörterbuch.")

        # Laden des Transformer-Modells
        self.transformer_core.load_state_dict(payload.pop("transformer_state_dict", {}))
        self.model = self.transformer_core
        self.transformer_core.to(self.device)

        # Laden des Optimierers
        optimizer_state_dict = payload.pop("optimizer_state_dict", None)
        if optimizer_state_dict:
            self.trainer.reset_optimizer(self.trainer.learning_rate)
            self.trainer.optimizer.load_state_dict(optimizer_state_dict)

        # Laden des Tokenizers (ggf. neu initialisieren)
        tokenizer_name = payload.pop("tokenizer_name", None)
        if tokenizer_name:
            # Annahme: Tokenizer kann mit einem Namen neu initialisiert werden
            self.tokenizer = tc.Tokenizer(pretrained_model_name_or_path=tokenizer_name)
            # Aktualisiere den Tokenizer im Trainer, falls dieser ihn auch hält
            self.trainer.tokenizer = self.tokenizer

        self.metrics = payload.pop("metrics", MetricsLogger())
        self._step_counter = int(payload.pop("step_counter", 0))
        self.ingested_text_data = payload.pop("ingested_text_data", [])

        # Falls importierte Metriken leer sind, aktuellen Zustand loggen, damit UI Werte hat.
        if not self.metrics.steps:
            current_lr = self.trainer.optimizer.param_groups[0]["lr"]
            self._log_now(0.0, 0.0, 0.0, current_lr)

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
    d_ff = d_model * 4
    max_seq_len = 512
    dropout = 0.1

    transformer_core = tc.TransformerCore(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
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
            new_transformer_core = tc.TransformerCore(
                vocab_size=new_tokenizer.vocab_size,
                d_model=d_model,
                num_layers=n_layers,
                num_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
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

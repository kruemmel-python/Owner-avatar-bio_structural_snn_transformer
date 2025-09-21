from __future__ import annotations
# ============================================================
# Spaced Repetition Trainer (SM-2-inspiriert) für dein Projekt
# - Angepasst für Transformer LLM
# - Wiederholungsplanung über "Steps" (keine Echtzeit nötig)
# - Nutzt Transformer-Metriken: perplexity, loss, embedding_similarity
# - Kontextvariation (leichter Jitter über Pseudo-Kontext)
# - Hoher Loss/Perplexity -> Intervallanpassung, gezieltes Re-Training
# - Konsolidierungs-/Replay-Blöcke nach Batches (simuliert)
# Nur Standardbibliothek, Python 3.12
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

# Dein neues Transformer-Modul:
import transformer_core as tc  # TransformerCore für LLM-Interaktion
import book_ingestion_vsa_adapter_plus as ap # Behalten für Text-Aufbereitung, aber nicht mehr für SNN-Logik


# --------------------------- Repräsentation -----------------------------

@dataclass
class RehearsalItem:
    tag: str                  # "chapXX_paraYY" (Absatz) oder "chapXX" (Kapitel)
    chapter_idx: int
    para_idx: Optional[int]   # None für Kapitel-Wiederholung
    text_content: str         # Der eigentliche Textinhalt für das LLM
    due_at_step: int = 0
    reps: int = 0
    interval: int = 20
    ease: float = 2.3         # SM-2-ähnliche „Leichtigkeit“


@dataclass
class ScheduleConfig:
    min_interval: int = 20
    max_interval: int = 10_000
    seed: int = 123
    # Qualitätsgewichtung für Transformer-Metriken
    w_loss: float = 5.0       # Hoher Loss ist schlecht
    w_sim: float = 4.0        # Hohe Ähnlichkeit ist gut


# ----------------------------- Scheduler -------------------------------

class SpacedScheduler:
    """Einfacher SM-2-inspirierter Planer auf 'Steps' statt Uhrzeit, angepasst für LLM-Metriken."""

    def __init__(self, cfg: ScheduleConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def _quality(self, loss: float, embedding_similarity: float) -> int:
        """
        0..5 (wie SM-2). Hoher Loss wird invers belohnt, hohe Ähnlichkeit direkt.
        Hart: sehr hoher Loss -> sehr schlechte Note.
        """
        # Annahme: Loss ist typischerweise > 0, Ähnlichkeit 0..1
        if loss > 1.5: # Sehr hoher Loss
            return 1
        if embedding_similarity < 0.6: # Geringe Ähnlichkeit
            return 2
        
        # Normalisiere Loss (Beispiel: 0.1 -> 1.0, 1.0 -> 0.0)
        # Hier ist eine Heuristik, die je nach typischem Loss-Bereich angepasst werden muss.
        # Angenommen, ein guter Loss ist < 0.5, ein schlechter > 1.0
        normalized_loss_score = max(0.0, 1.0 - min(1.0, (loss - 0.1) / 0.9)) # Skaliert Loss von 0.1-1.0 auf 1.0-0.0
        
        score = self.cfg.w_sim * max(0.0, min(1.0, embedding_similarity)) + self.cfg.w_loss * normalized_loss_score
        
        if score > 7.5:
            return 5
        if score > 6.5:
            return 4
        if score > 5.5:
            return 3
        return 2

    def seed_interval(self, policy: str) -> int:
        base = self.cfg.min_interval
        match policy:
            case "ebbinghaus":
                # grobe Stufen: kurz / mittel / länger
                return self.rng.choice([base, base * 10, base * 60])
            case "fixed":
                return max(base, 200)
            case "adaptive" | _:
                return base

    def update(self, item: RehearsalItem, loss: float, embedding_similarity: float, now_step: int) -> None:
        q = self._quality(loss, embedding_similarity)
        if q < 3:
            item.reps = 0
            item.interval = max(self.cfg.min_interval, item.interval // 2)
            item.ease = max(1.3, item.ease - 0.2)
        else:
            item.reps += 1
            if item.reps == 1:
                item.interval = max(self.cfg.min_interval, item.interval)
            elif item.reps == 2:
                item.interval = max(self.cfg.min_interval, int(item.interval * 2.5))
            else:
                item.ease = min(2.8, item.ease + 0.05)
                item.interval = min(self.cfg.max_interval, int(item.interval * item.ease))
        jitter = int(0.1 * item.interval)
        item.due_at_step = now_step + item.interval + self.rng.randint(-jitter, +jitter)


# ------------------------------- Trainer -------------------------------

@dataclass
class Trainer:
    # Der Adapter wird nur noch für die Text-Aufbereitung genutzt, nicht mehr für SNN-Logik
    adapter: ap.BookAdapter
    # Das Transformer-Modell ist der neue Kern
    transformer_model: tc.TransformerCore = field(default_factory=tc.TransformerCore)
    scheduler: SpacedScheduler = field(default_factory=lambda: SpacedScheduler(ScheduleConfig()))
    replay_after_n_items: int = 6         # Batchgröße für Konsolidierungs-/Re-Training-Block
    ctx_jitter_strength: int = 2          # kleine Kontextvariation (für Transformer-Eingabe)
    max_steps: int = 50_000               # Schutz, falls jemand unendliche Epochen setzt
    log: List[Dict[str, float]] = field(default_factory=list)

    # ---- Hilfen (Metriken/Stimuli) ----
    def _metrics(self) -> Tuple[float, float, float, int]:
        # Metriken kommen jetzt vom Transformer-Modell
        current_loss = self.transformer_model.get_current_loss()
        current_perplexity = self.transformer_model.get_current_perplexity()
        avg_embedding_similarity = self.transformer_model.get_average_embedding_similarity()
        # Anzahl der trainierten Token/Batches könnte eine Metrik sein
        trained_steps = self.transformer_model.get_trained_steps()
        return current_loss, current_perplexity, avg_embedding_similarity, trained_steps

    def _log_now(self, step: int) -> None:
        loss, perplexity, sim, trained_steps = self._metrics()
        self.log.append({
            "step": float(step),
            "loss": float(loss),
            "perplexity": float(perplexity),
            "embedding_similarity": float(sim),
            "trained_steps": float(trained_steps),
        })

    def _paragraphs(self) -> List[Tuple[str, int, int, str]]:
        """
        Liefert Liste (tag, chapter_idx, para_idx, text_content).
        Greift auf die öffentliche Struktur des Adapters zu.
        """
        out: List[Tuple[str, int, int, str]] = []
        for p in getattr(self.adapter, "_paragraphs", []):
            out.append((p.tag, p.chapter_idx, p.paragraph_idx, p.text_content))
        return out

    def _train_on_item(self, item: RehearsalItem) -> Tuple[float, float]:
        """
        Trainiert das Transformer-Modell auf dem Textinhalt des RehearsalItem.
        Gibt Loss und Embedding-Ähnlichkeit zurück.
        """
        # Hier könnte man auch Kontext-Jitter anwenden, z.B. durch leicht veränderte Prompts
        # oder durch Hinzufügen/Entfernen von Sätzen aus dem umgebenden Kontext.
        # Für den Anfang trainieren wir direkt auf dem Absatz.
        
        # Der `train_on_text` Methode des TransformerCore wird der Text übergeben.
        # Sie gibt Loss und eine Metrik für die Qualität des gelernten Embeddings zurück.
        loss, embedding_similarity = self.transformer_model.train_on_text(item.text_content)
        return loss, embedding_similarity

    def _consolidate_and_retrain(self, rounds: int = 2) -> None:
        """
        Simuliert Konsolidierung durch gezieltes Re-Training auf "schwierigen" oder
        "wichtigen" Inhalten.
        """
        # Hier könnte man z.B. die Items mit dem höchsten Loss oder den längsten Intervallen
        # für ein zusätzliches Re-Training auswählen.
        # Für diese Implementierung rufen wir eine generische Konsolidierungsfunktion auf.
        self.transformer_model.consolidate_knowledge(rounds=rounds)

    # ---- Öffentliches API ----
    def build_queue(self, include_chapter_nodes: bool = True) -> List[RehearsalItem]:
        q: List[RehearsalItem] = []
        seen_ch: set[int] = set()
        for tag, ch, pi, text_content in self._paragraphs():
            q.append(RehearsalItem(tag=tag, chapter_idx=ch, para_idx=pi, text_content=text_content))
            seen_ch.add(ch)
        if include_chapter_nodes:
            # Für Kapitel-Wiederholung könnte man eine Zusammenfassung oder den ersten Absatz nehmen
            # Hier nehmen wir einfach den Text des ersten Absatzes des Kapitels als Repräsentation
            for ch in sorted(seen_ch):
                chapter_text = ""
                for item in q:
                    if item.chapter_idx == ch and item.para_idx == 0: # Erster Absatz des Kapitels
                        chapter_text = item.text_content
                        break
                q.append(RehearsalItem(tag=f"chap{ch:02d}", chapter_idx=ch, para_idx=None, text_content=chapter_text))
        return q

    def repeat(self, epochs: int = 3, policy: str = "ebbinghaus") -> None:
        """
        Führt echtes Wiederholen durch. Jede Epoche:
         - Pick das jeweils fällige Item
         - Trainiere das LLM auf dem Absatz (mit simuliertem Kontext-Jitter)
         - Messe Loss/Embedding-Ähnlichkeit -> Update Intervall
         - Jede N Items: „Konsolidierung“ (Re-Training)
        """
        queue = self.build_queue(include_chapter_nodes=True)
        for it in queue:
            it.interval = self.scheduler.seed_interval(policy)
            it.due_at_step = it.interval

        items_since_consolidation = 0
        step = 0
        self._log_now(step)

        for _ in range(max(1, epochs)):
            while step < self.max_steps:
                due = [it for it in queue if it.due_at_step <= step]
                if not due:
                    step += 10
                    if step % 100 == 0:
                        self._log_now(step)
                    continue

                due.sort(key=lambda it: it.due_at_step)
                item = due[0]

                # Trainiere das Transformer-Modell auf dem aktuellen Item
                loss, embedding_similarity = self._train_on_item(item)

                # Wenn Loss hoch oder Ähnlichkeit niedrig, triggere sofortiges Re-Training
                if loss > 1.0 or embedding_similarity < 0.7:
                    # Gezieltes Re-Training auf diesem Item
                    self.transformer_model.train_on_text(item.text_content, epochs=2) # Zusätzliche Epochen

                self.scheduler.update(item, loss=loss, embedding_similarity=embedding_similarity, now_step=step)

                items_since_consolidation += 1
                if items_since_consolidation >= self.replay_after_n_items:
                    self._consolidate_and_retrain(rounds=2)
                    items_since_consolidation = 0

                step += max(5, item.interval // 4) # Fortschritt im "Trainings-Schritt"
                self._log_now(step)

                # Abbruchkriterium: Wenn alle Intervalle sehr lang sind, ist das Wissen gut konsolidiert
                if all(it.interval > 2000 for it in queue):
                    break

            # Nach jeder "Epoche" eine weitere Konsolidierungsphase
            self._consolidate_and_retrain(rounds=5)
            step = 0 # Reset des Schrittzählers für die nächste Epoche, oder weiterlaufen lassen?
                     # Für SM-2-Ähnlichkeit ist ein Reset pro Epoche sinnvoll, um "fällige" Items neu zu bewerten.
            self._log_now(step)


from __future__ import annotations

# ------------------------------------------------------------
# Transformer-Kern (ersetzt den biologisch inspirierten SNN-Kern)
# - Dieses Modul dient nun als Schnittstelle zum eigentlichen Transformer-Modell.
# - Die detaillierte Implementierung des Transformer-Modells und seiner Trainingslogik
#   wird in einem separaten Modul (z.B. transformer_core.py) behandelt.
# - Hier werden die grundlegenden Funktionen für die Interaktion mit dem Transformer
#   bereitgestellt, wie z.B. das Verarbeiten von Stimuli, Kontext, Belohnungen
#   und das Auslösen von "Replay"- oder Konsolidierungsphasen.
# ------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
from enum import Enum, auto
import time
import random
import math

# Import des zukünftigen Transformer-Kerns
# Dies ist ein Platzhalter und muss später durch die tatsächliche Implementierung ersetzt werden.
# from .transformer_core import TransformerModel, TransformerConfig, TransformerTrainer

# ==============================
# 0) Ereignisse & Utils
# ==============================

class EventKind(Enum):
    STIMULUS = auto()   # Input für den Transformer (z.B. tokenisierte Textsequenz)
    CTX      = auto()   # Kontext-Input (z.B. Embedding eines Kontextvektors)
    REWARD   = auto()   # Belohnung (für Reinforcement Learning oder Dopamin-Modulation)
    SLEEP    = auto()   # Konsolidierungs-/Replay-Phase des Transformers
    TICK     = auto()   # freier Zeitschritt (für interne Zustandsaktualisierungen)

@dataclass
class Event:
    kind: EventKind
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ==============================
# 1) Hyperparameter (angepasst für Transformer-Interaktion)
# ==============================

@dataclass
class Hyper:
    # Transformer-Modell-Parameter (Platzhalter, Details in transformer_core.py)
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 6
    transformer_seq_len: int = 256
    transformer_lr: float = 1e-4
    transformer_dropout: float = 0.1

    # Input/Output-Dimensionen (Anpassung an VSA-Adapter und Kontext)
    input_embedding_dim: int = 512  # Dimension der Eingabe-Embeddings
    context_embedding_dim: int = 60 # Dimension des Kontext-Embeddings

    # Belohnung / Dopamin-Modulation
    dopamine_decay: float = 0.98
    dopamine_gain: float = 0.8

    # Replay / Konsolidierung
    replay_batch_size: int = 4
    replay_steps_per_round: int = 10
    replay_temp_base: float = 1.0
    replay_temp_max: float = 1.8
    replay_temp_min: float = 0.6
    feedback_to_temp_factor: float = 0.8 # Wie stark Feedback die Replay-Temperatur beeinflusst

    # Comparator-Feedback (Metrik-basiert, nicht mehr neuronales Mismatch)
    mismatch_threshold: float = 0.05 # Ab diesem normalisierten Level "Hard-Gate" auslösen
    hard_gate_cooldown_steps: int = 20
    hard_gate_learning_reduction: float = 0.2 # Wie stark das Lernen im Hard-Gate reduziert wird
    hard_gate_temp_boost: float = 1.15

    # Phasen-spezifische Replay-Politiken (konzeptionell für Transformer)
    # Könnten sich auf Lernraten, Dropout oder Temperatur auswirken
    replay_potent_lr_factor: float = 0.8    # Im "Potenzfenster": niedrigere LR (mehr Konsolidierung)
    replay_anti_lr_factor: float = 1.2      # Außerhalb: höhere LR (mehr Exploration)
    replay_potent_dropout_factor: float = 0.9
    replay_anti_dropout_factor: float = 1.1

# ==============================
# 2) HippocampalSystem (als Schnittstelle zum Transformer)
# ==============================

@dataclass
class Engram:
    """
    Repräsentiert ein "Engramm" oder eine gespeicherte Erfahrung für das Replay.
    Für den Transformer könnte dies eine Sequenz von Token-IDs oder Embeddings sein,
    zusammen mit dem Kontext, in dem sie gelernt wurde.
    """
    stimulus_data: List[int] # Z.B. Token-IDs
    context_data: List[float] # Z.B. Kontext-Embedding
    tag: str = ""
    saliency: float = 1.0
    ts: float = field(default_factory=time.time)

class HippocampalSystem:
    def __init__(self, h: Optional[Hyper] = None, seed: Optional[int] = 7):
        self.h = h or Hyper()
        if seed is not None: random.seed(seed)

        # Platzhalter für den Transformer-Modell-Kern
        # In einer realen Implementierung würde hier ein TransformerModel instanziiert.
        # self.transformer_model = TransformerModel(TransformerConfig(...))
        # self.transformer_trainer = TransformerTrainer(self.transformer_model, self.h.transformer_lr)
        print("WARNUNG: Transformer-Modell ist noch ein Platzhalter. Echte Implementierung in transformer_core.py erwartet.")

        # Interner Zustand
        self.current_stimulus_embedding: Optional[List[float]] = None
        self.current_context_embedding: Optional[List[float]] = None
        self.dopamine_level: float = 0.0
        self.engrams: List[Engram] = []
        self.sleeping: bool = False
        self.replay_temp: float = self.h.replay_temp_base
        self.hard_gate_countdown: int = 0
        self.current_mismatch_metric: float = 0.0 # Eine Metrik für den Unterschied zwischen Erwartung und Realität

        # Für die Simulation der Phasen-Politik (ersetzt den Oszillator)
        self._internal_tick: int = 0
        self.theta_period: int = 12 # Simuliert eine Theta-Phase
        self.theta_potent_window: Tuple[float, float] = (0.2, 0.6)

    # --- Interne Hilfsfunktionen ---
    def _get_theta_phase(self) -> float:
        """Simuliert eine Theta-Phase basierend auf internem Tick."""
        return (self._internal_tick % self.theta_period) / self.theta_period

    def _in_potent_window(self) -> bool:
        """Prüft, ob sich das System in einem "Potenzierungsfenster" befindet."""
        th = self._get_theta_phase()
        t0, t1 = self.theta_potent_window
        return t0 <= th <= t1

    def _update_dopamine(self, magnitude: float) -> None:
        """Aktualisiert den Dopaminspiegel."""
        self.dopamine_level = min(1.0, self.dopamine_level * self.h.dopamine_decay + magnitude)

    def _calculate_mismatch(self) -> float:
        """
        Berechnet eine Mismatch-Metrik.
        In einem echten Transformer-System könnte dies z.B. der Loss,
        die Divergenz zwischen Vorhersage und Ziel oder eine andere
        Fehlermetrik sein.
        Dies ist ein Platzhalter.
        """
        # Beispiel: Zufälliger Mismatch für Demo-Zwecke
        return random.uniform(0.01, 0.1)

    def _apply_feedback_to_learning_params(self) -> None:
        """
        Passt Transformer-Lernparameter basierend auf Feedback und Hard-Gate an.
        """
        # Grundlegende Anpassung der Replay-Temperatur basierend auf Mismatch
        temp_base = self.h.replay_temp_base + self.h.feedback_to_temp_factor * self.current_mismatch_metric
        self.replay_temp = clamp(max(self.replay_temp, temp_base), self.h.replay_temp_min, self.h.replay_temp_max)

        # Hard-Gate Logik
        if self.current_mismatch_metric >= self.h.mismatch_threshold:
            self.hard_gate_countdown = max(self.hard_gate_countdown, self.h.hard_gate_cooldown_steps)

        current_learning_reduction = 1.0
        if self.hard_gate_countdown > 0:
            current_learning_reduction = self.h.hard_gate_learning_reduction
            self.replay_temp = clamp(self.replay_temp * self.h.hard_gate_temp_boost,
                                     self.h.replay_temp_min, self.h.replay_temp_max)
            self.hard_gate_countdown -= 1

        # Hier würden die Lernraten, Dropout etc. des Transformers angepasst
        # self.transformer_trainer.set_learning_rate_multiplier(current_learning_reduction)
        # self.transformer_model.set_dropout_multiplier(1.0 / current_learning_reduction)
        # print(f"DEBUG: Hard-Gate aktiv: Lernreduktion={current_learning_reduction:.2f}, Replay-Temp={self.replay_temp:.2f}")

    # --- Public I/O ---
    def present_stimulus(self, stimulus_embedding: List[float]) -> None:
        """
        Präsentiert einen Stimulus (z.B. ein VSA-Embedding eines Wortes/Satzes)
        an den Transformer.
        """
        self.current_stimulus_embedding = stimulus_embedding
        # Hier würde der Stimulus an den Transformer übergeben und verarbeitet
        # z.B. self.transformer_model.process_input(stimulus_embedding, self.current_context_embedding)

    def present_context(self, context_embedding: List[float]) -> None:
        """
        Präsentiert einen Kontext (z.B. ein VSA-Embedding eines Kontextvektors)
        an den Transformer.
        """
        self.current_context_embedding = context_embedding
        # Der Kontext würde intern im Transformer für Gating oder Konditionierung verwendet
        # z.B. self.transformer_model.set_context(context_embedding)

    def step(self) -> None:
        """
        Führt einen Simulationsschritt aus.
        In einem Transformer-System könnte dies eine Vorwärts- und Rückwärtspass-Iteration
        oder einfach eine Zustandsaktualisierung sein.
        """
        self._internal_tick += 1

        # Aktualisiere Dopamin-Trace
        self.dopamine_level *= self.h.dopamine_decay

        # Führe einen Schritt im Transformer aus (Platzhalter)
        # output = self.transformer_model.forward(self.current_stimulus_embedding, self.current_context_embedding)
        # self.current_mismatch_metric = self._calculate_mismatch_from_output(output) # Echte Mismatch-Berechnung

        # Für Demo-Zwecke: Simuliere Mismatch
        self.current_mismatch_metric = self._calculate_mismatch()

        # Wende Feedback auf Lernparameter an
        self._apply_feedback_to_learning_params()

        # Optional: Leere aktuelle Inputs nach Verarbeitung
        self.current_stimulus_embedding = None
        self.current_context_embedding = None

    def reward(self, magnitude: float = 0.5) -> None:
        """
        Gibt eine Belohnung, die den Dopaminspiegel erhöht.
        """
        self._update_dopamine(magnitude)
        # Die Belohnung könnte auch direkt in die RL-Komponente des Transformers einfließen
        # self.transformer_trainer.apply_reward(magnitude)

    def store_engram(self, stimulus_data: List[int], context_data: List[float], tag: str = "", saliency: float = 1.0) -> None:
        """
        Speichert eine Erfahrung (Engramm) für zukünftiges Replay.
        """
        self.engrams.append(Engram(stimulus_data=stimulus_data, context_data=context_data, tag=tag, saliency=saliency))

    def _softmax_temp(self, xs: List[float], temp: float) -> List[float]:
        if not xs: return []
        x = [v / max(1e-9, temp) for v in xs]
        m = max(x)
        exps = [math.exp(v - m) for v in x]
        s = sum(exps)
        return [v / s for v in exps]

    def sleep_replay(self, rounds: int = 2) -> None:
        """
        Führt eine Konsolidierungs-/Replay-Phase durch.
        Der Transformer würde hier mit gespeicherten Engrammen trainiert.
        """
        if not self.engrams:
            # print("Keine Engramme für Replay vorhanden.")
            return

        self.sleeping = True
        # print(f"Starte Replay-Phase für {rounds} Runden...")

        for _ in range(rounds):
            # Auswahl der Engramme mit Temperatur-basierter Softmax-Verteilung
            now = time.time()
            scores = []
            for engram in self.engrams:
                recency_factor = 1.0 / (1.0 + (now - engram.ts))
                scores.append(max(1e-6, engram.saliency * recency_factor))

            probs = self._softmax_temp(scores, temp=self.replay_temp)
            k = min(self.h.replay_batch_size, len(self.engrams))
            picks = random.choices(self.engrams, weights=probs, k=k)

            for engram in picks:
                for step_i in range(self.h.replay_steps_per_round):
                    # Phasen-spezifische Replay-Politik (konzeptionell für Transformer)
                    in_potent = self._in_potent_window()
                    current_lr_factor = 1.0
                    current_dropout_factor = 1.0

                    if in_potent:
                        # Konsolidierendes Replay: niedrigere LR, weniger Dropout
                        current_lr_factor = self.h.replay_potent_lr_factor
                        current_dropout_factor = self.h.replay_potent_dropout_factor
                    else:
                        # Exploratives Replay: höhere LR, mehr Dropout
                        current_lr_factor = self.h.replay_anti_lr_factor
                        current_dropout_factor = self.h.replay_anti_dropout_factor

                    # Hier würden die Transformer-Lernparameter für diesen Replay-Schritt angepasst
                    # self.transformer_trainer.set_learning_rate_multiplier(current_lr_factor)
                    # self.transformer_model.set_dropout_multiplier(current_dropout_factor)

                    # Führe einen Trainingsschritt mit dem Engramm im Transformer aus
                    # self.transformer_trainer.train_on_engram(engram.stimulus_data, engram.context_data)
                    # Simuliere einen Schritt
                    self.step()

        self.sleeping = False
        # print("Replay-Phase beendet.")

    # --- Ereignisse ---
    def deliver(self, ev: Event) -> None:
        """
        Verarbeitet ein externes Ereignis.
        """
        match ev.kind:
            case EventKind.STIMULUS:
                # Annahme: ev.data["embedding"] enthält ein vorbereitetes Embedding
                embedding = list(ev.data.get("embedding", []))
                if not embedding:
                    raise ValueError("STIMULUS-Ereignis benötigt 'embedding'-Daten.")
                self.present_stimulus(embedding)
                self.step()
            case EventKind.CTX:
                # Annahme: ev.data["embedding"] enthält ein vorbereitetes Kontext-Embedding
                embedding = list(ev.data.get("embedding", []))
                if not embedding:
                    raise ValueError("CTX-Ereignis benötigt 'embedding'-Daten.")
                self.present_context(embedding)
                self.step()
            case EventKind.REWARD:
                self.reward(float(ev.data.get("magnitude", 0.5)))
            case EventKind.SLEEP:
                self.sleep_replay(int(ev.data.get("rounds", 3)))
            case EventKind.TICK:
                self.step()
            case _:
                raise ValueError(f"Unbekanntes Ereignis: {ev.kind!r}")

# ==============================
# 3) Demo / Tests (angepasst für Transformer-Schnittstelle)
# ==============================

def generate_random_embedding(dim: int) -> List[float]:
    """Generiert ein zufälliges Embedding."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]

def generate_random_token_ids(seq_len: int, vocab_size: int) -> List[int]:
    """Generiert zufällige Token-IDs."""
    return [random.randint(0, vocab_size - 1) for _ in range(seq_len)]

def demo() -> None:
    h = Hyper()
    sys = HippocampalSystem(h, seed=31)

    # Simuliere Stimuli und Kontexte als Embeddings
    # In einer echten Anwendung kämen diese vom book_ingestion_vsa_adapter_plus.py
    A1_tokens = generate_random_token_ids(h.transformer_seq_len, 10000) # Beispiel: Token-IDs
    A2_tokens = generate_random_token_ids(h.transformer_seq_len, 10000)
    A3_tokens = generate_random_token_ids(h.transformer_seq_len, 10000)

    C1_embedding = generate_random_embedding(h.context_embedding_dim)
    C2_embedding = generate_random_embedding(h.context_embedding_dim)

    print("Starte Lernphase...")
    # Lernen (mit Kontext)
    for _ in range(3):
        sys.deliver(Event(EventKind.CTX,      data={"embedding": C1_embedding}))
        sys.deliver(Event(EventKind.STIMULUS, data={"embedding": generate_random_embedding(h.input_embedding_dim)}))
        for _ in range(2): sys.deliver(Event(EventKind.TICK))
    sys.store_engram(A1_tokens, C1_embedding, "A1", saliency=1.4)

    for _ in range(3):
        sys.deliver(Event(EventKind.CTX,      data={"embedding": C2_embedding}))
        sys.deliver(Event(EventKind.STIMULUS, data={"embedding": generate_random_embedding(h.input_embedding_dim)}))
        for _ in range(2): sys.deliver(Event(EventKind.TICK))
    sys.store_engram(A2_tokens, C2_embedding, "A2", saliency=1.0)

    for _ in range(3):
        sys.deliver(Event(EventKind.CTX,      data={"embedding": C1_embedding}))
        sys.deliver(Event(EventKind.STIMULUS, data={"embedding": generate_random_embedding(h.input_embedding_dim)}))
        for _ in range(2): sys.deliver(Event(EventKind.TICK))
    sys.store_engram(A3_tokens, C1_embedding, "A3", saliency=0.8)

    print("\nBelohnung vergeben...")
    sys.deliver(Event(EventKind.REWARD, data={"magnitude": 0.6}))

    print("\nStarte Schlafphase (Replay)...")
    sys.deliver(Event(EventKind.SLEEP, data={"rounds": 3}))

    print("\nEvaluationsphase (simuliert)...")
    def evaluate_system(ctx_embedding: List[float]) -> Tuple[float, float, int]:
        sys.deliver(Event(EventKind.CTX, data={"embedding": ctx_embedding}))
        # Simuliere eine Abfrage oder einen weiteren Stimulus
        sys.deliver(Event(EventKind.STIMULUS, data={"embedding": generate_random_embedding(h.input_embedding_dim)}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
        # Hier würden Metriken vom Transformer abgefragt werden
        # z.B. Vorhersagegüte, Kohärenz des Outputs etc.
        # Für die Demo nutzen wir die internen Mismatch- und Replay-Temp-Werte
        return sys.current_mismatch_metric, sys.replay_temp, sys.hard_gate_countdown

    mismatch1, temp1, hg1 = evaluate_system(C1_embedding)
    mismatch2, temp2, hg2 = evaluate_system(C2_embedding)

    print(f"Mismatch (C1): {mismatch1:.4f} | Replay-Temp (C1): {temp1:.2f} | HardGate-Countdown (C1): {hg1}")
    print(f"Mismatch (C2): {mismatch2:.4f} | Replay-Temp (C2): {temp2:.2f} | HardGate-Countdown (C2): {hg2}")
    print(f"Aktueller Dopaminspiegel: {sys.dopamine_level:.4f}")
    print(f"Anzahl gespeicherter Engramme: {len(sys.engrams)}")

if __name__ == "__main__":
    demo()

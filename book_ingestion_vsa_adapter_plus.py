from __future__ import annotations
"""Vereinfachter Bucheingabe-Adapter zur Vorbereitung von Textdaten für ein Transformer-Modell.

Dieses Modul stellt die ``BookAdapter``-Klasse bereit, die für die Streamlit-UI und die CLI-Instrumentierung
benötigt wird. Das Ziel ist es, eine leichtgewichtige, eigenständige Schnittstelle zu schaffen,
die Textdaten für ein nachgeschaltetes Transformer-Modell vorverarbeitet.

Die Implementierung konzentriert sich auf die Tokenisierung und die Erzeugung von Embeddings
für Sätze und Absätze, anstatt bio-inspirierte SNN-Verhaltensweisen zu emulieren.

* Sätze werden mit einem Transformer-Tokenizer tokenisiert.
* Embeddings werden für Sätze und Absätze generiert.
* Die Adapter-Methoden werden so angepasst, dass sie die Transformer-Logik widerspiegeln
  oder als Platzhalter für zukünftige Integrationen dienen.
"""

from dataclasses import dataclass, field
import math
import pickle
import random
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Importiere Transformer-spezifische Bibliotheken
# Annahme: Diese sind im System verfügbar oder werden später hinzugefügt.
# Für dieses Shim verwenden wir Platzhalter, die die Struktur andeuten.
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    # Lade ein kleines, schnelles Modell für die Tokenisierung und Embeddings
    # Dies ist ein Platzhalter und sollte durch ein geeignetes Modell ersetzt werden.
    _TRANSFORMER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _tokenizer = AutoTokenizer.from_pretrained(_TRANSFORMER_MODEL_NAME)
    _model = AutoModel.from_pretrained(_TRANSFORMER_MODEL_NAME)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)
except ImportError:
    print("Warnung: Hugging Face Transformers nicht installiert. Der BookAdapter wird im Mock-Modus ausgeführt.")
    _tokenizer = None
    _model = None
    _device = "cpu"


# Hilfsfunktion zur Generierung von Mock-Embeddings, wenn Transformers nicht verfügbar sind
def _mock_generate_embedding(text: str) -> List[float]:
    # Eine einfache Hash-basierte "Embedding"-Generierung für den Mock-Modus
    # In einer echten Implementierung würde hier ein Transformer-Modell verwendet.
    random.seed(hash(text) % (2**32 - 1))
    return [random.uniform(-1.0, 1.0) for _ in range(384)] # Beispiel-Dimension


def _generate_embedding(text: str) -> List[float]:
    if _tokenizer and _model:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(_device)
        with torch.no_grad():
            outputs = _model(**inputs)
        # Verwende den Mittelwert der letzten Hidden States als Satz-Embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        return embedding
    else:
        return _mock_generate_embedding(text)


TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def _normalise_word(word: str) -> str:
    return word.lower()


def _sentence_tokens(sentence: str) -> List[str]:
    # Für die Transformer-Pipeline ist die Tokenisierung des Modells relevanter,
    # aber diese Funktion wird noch für die Kompatibilität und einfache Zählungen beibehalten.
    return [_normalise_word(tok) for tok in TOKEN_RE.findall(sentence)]


@dataclass
class EncodedSentence:
    """Container, der die Struktur für die Transformer-Pipeline widerspiegelt."""

    text: str
    tokens: List[str] # Einfache Wort-Tokens
    embedding: List[float] = field(default_factory=list) # Transformer-Embedding
    paragraph_tag: str = ""


class SimpleCortex:
    """Platzhalter für die Kohärenzbewertung, jetzt basierend auf Embeddings."""

    def __init__(self, memory: int = 32) -> None:
        self._history: List[List[float]] = [] # Speichert Embeddings statt Token-Counter
        self._memory = memory

    def observe(self, embedding: List[float]) -> None:
        self._history.append(embedding)
        if len(self._history) > self._memory:
            self._history.pop(0)

    def coherence_score(self) -> float:
        if len(self._history) < 2:
            return 1.0 if self._history else 0.0
        # Durchschnittliche paarweise Kosinusähnlichkeit über das Verlaufsfenster.
        sims = []
        for i in range(1, len(self._history)):
            sims.append(_cosine_embedding(self._history[i - 1], self._history[i]))
        if not sims:
            return 0.0
        return sum(sims) / len(sims)


def _cosine_embedding(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Paragraph:
    tag: str
    chapter_idx: int
    paragraph_idx: int
    text: str
    sentences: List[EncodedSentence]
    embedding: List[float] = field(default_factory=list) # Embedding für den gesamten Absatz


def _chunk(seq: Sequence[str], size: int) -> List[List[str]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _ensure_attr(obj: object, name: str, default):
    if not hasattr(obj, name):
        setattr(obj, name, default)
    return getattr(obj, name)


class BookAdapter:
    """Minimale, aber funktionale Bucheingabe-Pipeline für Transformer-Modelle.

    Dieser Adapter bereitet Textdaten für ein Transformer-Modell vor und ersetzt
    die bio-inspirierte SNN-Logik durch Transformer-zentrierte Konzepte.
    """

    def __init__(self, system: Optional[object] = None) -> None:
        self.system = system or object()
        self.cortex = SimpleCortex() # Jetzt mit Embedding-basierter Kohärenz
        self._paragraphs: List[Paragraph] = []
        self._last_embedding: List[float] = [] # Speichert das Embedding des letzten Satzes
        self._rng = random.Random(1234)
        self._chapter_titles: Dict[int, str] = {}
        self._paragraph_lookup: Dict[str, Paragraph] = {}
        # _token_mastery wird für Transformer-Modelle weniger relevant,
        # da Familiarität eher über Embeddings oder Modellzustände abgeleitet wird.
        # Wir behalten es als Platzhalter bei, falls es für Metriken nützlich ist.
        self._token_mastery: Counter[str] = Counter()

        # Sicherstellen, dass die nachgeschaltete Instrumentierung die erwarteten Felder findet.
        # Diese Felder sind jetzt Platzhalter und spiegeln nicht mehr direkt SNN-Zustände wider.
        _ensure_attr(self.system, "ca1_last_mismatch", [0.0])
        _ensure_attr(self.system, "engrams", [])
        _ensure_attr(self.system, "replay_temp", 1.0)
        _ensure_attr(self.system, "hard_gate_countdown", 0)
        _ensure_attr(self.system, "average_familiarity", 0.0)

    # ------------------------------------------------------------------
    # High level ingestion API
    # ------------------------------------------------------------------
    def ingest_book(
        self,
        text: str,
        chapter_size_sents: int = 40,
        para_size_sents: int = 6,
        *,
        reset_state: bool = True,
    ) -> None:
        sentences = [s.strip() for s in re.split(r"[\n\.\?!]+", text) if s.strip()]
        if not sentences:
            return

        chapters = _chunk(sentences, chapter_size_sents)
        if reset_state:
            timestamp = time.time()
            self._paragraphs = []
            self._paragraph_lookup = {}
            self._chapter_titles = {}
            self.system.engrams.clear()
            self.cortex = SimpleCortex()
            self._last_embedding = []
            self._token_mastery = Counter()
            base_chapter_idx = 0
        else:
            last_ts = (
                max((eng.get("ts", 0.0) for eng in getattr(self.system, "engrams", [])), default=0.0)
                if getattr(self.system, "engrams", None)
                else 0.0
            )
            timestamp = max(time.time(), last_ts + 1.0)
            if self._chapter_titles:
                base_chapter_idx = max(self._chapter_titles.keys()) + 1
            else:
                base_chapter_idx = 0

        for ch_idx, chapter in enumerate(chapters):
            global_ch_idx = base_chapter_idx + ch_idx
            chapter_title = chapter[0] if chapter else ""
            chapter_title = chapter_title.strip()
            if chapter_title:
                preview = chapter_title[:80]
                self._chapter_titles[global_ch_idx] = f"Kapitel {global_ch_idx + 1}: {preview}"
            else:
                self._chapter_titles[global_ch_idx] = f"Kapitel {global_ch_idx + 1}"
            paragraphs = _chunk(chapter, para_size_sents)
            for para_idx, para_sentences in enumerate(paragraphs):
                tag = f"chap{global_ch_idx:02d}_para{para_idx:02d}"
                encoded_sentences: List[EncodedSentence] = []
                paragraph_full_text = " ".join(para_sentences)
                paragraph_embedding = _generate_embedding(paragraph_full_text)

                for sentence in para_sentences:
                    tokens = _sentence_tokens(sentence) # Für Kompatibilität
                    embedding = _generate_embedding(sentence)
                    enc = EncodedSentence(
                        text=sentence,
                        tokens=tokens,
                        embedding=embedding,
                        paragraph_tag=tag,
                    )
                    self._stimulate_sentence(enc) # Angepasster Stimulations-Hook
                    encoded_sentences.append(enc)

                paragraph = Paragraph(
                    tag=tag,
                    chapter_idx=ch_idx,
                    paragraph_idx=para_idx,
                    text=paragraph_full_text,
                    sentences=encoded_sentences,
                    embedding=paragraph_embedding,
                )
                self._paragraphs.append(paragraph)
                self._paragraph_lookup[tag] = paragraph
                self.system.engrams.append(
                    {
                        "ids": list(range(len(encoded_sentences))),
                        "tag": tag,
                        "saliency": self._engram_saliency(), # Platzhalter
                        "ts": timestamp,
                    }
                )
                timestamp += 1.0
            self.compress_chapter(global_ch_idx) # Platzhalter
            self.monitor_and_intervene(global_ch_idx) # Platzhalter

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _rebuild_lookup(self) -> None:
        self._paragraph_lookup = {para.tag: para for para in self._paragraphs}

    def export_state_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "cortex": self.cortex,
            "paragraphs": self._paragraphs,
            "chapter_titles": self._chapter_titles,
            "last_embedding": self._last_embedding,
            "token_mastery": self._token_mastery, # Beibehalten als Platzhalter
            "rng_state": self._rng.getstate(),
        }

    def import_state_dict(self, payload: Dict[str, Any]) -> None:
        self.system = payload.get("system", self.system)
        self.cortex = payload.get("cortex", SimpleCortex())
        self._paragraphs = payload.get("paragraphs", [])
        self._chapter_titles = payload.get("chapter_titles", {})
        self._last_embedding = payload.get("last_embedding", [])
        self._token_mastery = payload.get("token_mastery", Counter())
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self._rebuild_lookup()
        # Sicherstellen, dass die nachgeschaltete Instrumentierung die erwarteten Felder findet.
        _ensure_attr(self.system, "ca1_last_mismatch", [0.0])
        _ensure_attr(self.system, "engrams", [])
        _ensure_attr(self.system, "replay_temp", 1.0)
        _ensure_attr(self.system, "hard_gate_countdown", 0)
        _ensure_attr(self.system, "average_familiarity", 0.0)

    def export_state_bytes(self) -> bytes:
        return pickle.dumps(self.export_state_dict(), protocol=pickle.HIGHEST_PROTOCOL)

    def import_state_bytes(self, data: bytes) -> None:
        payload = pickle.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Ungültiges Gehirnformat: Erwartet Wörterbuch.")
        self.import_state_dict(payload)

    # ------------------------------------------------------------------
    # Interactive helpers mirroring the original adapter
    # ------------------------------------------------------------------
    def recall_paragraph_like(self, query: str, topk: int = 5) -> List[Tuple[str, float]]:
        query_embedding = _generate_embedding(query)
        if not query_embedding:
            return []
        scored = []
        for para in self._paragraphs:
            score = _cosine_embedding(query_embedding, para.embedding)
            if score > 0.0:
                scored.append((para.tag, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    def get_paragraph_metadata(self, tag: str) -> Optional[Dict[str, object]]:
        """Gibt beschreibende Metadaten für einen gespeicherten Absatz-Tag zurück."""

        paragraph = self._paragraph_lookup.get(tag)
        if not paragraph:
            return None
        chapter_label = self._chapter_titles.get(
            paragraph.chapter_idx, f"Kapitel {paragraph.chapter_idx + 1}"
        )
        preview = paragraph.text.strip()
        if len(preview) > 240:
            preview = preview[:237].rstrip() + "…"
        return {
            "tag": paragraph.tag,
            "chapter_index": paragraph.chapter_idx,
            "chapter_label": chapter_label,
            "paragraph_index": paragraph.paragraph_idx,
            "text": preview,
        }

    def ask_relation(self, subj: str, verb: str, obj: str) -> Tuple[int, int, float]:
        # Diese Funktion ist stark an die alte Token-basierte Logik gebunden.
        # Für ein Transformer-Modell müsste dies durch eine semantische Abfrage
        # oder ein spezialisiertes Relation-Extraction-Modell ersetzt werden.
        # Hier wird eine vereinfachte, Token-basierte Logik beibehalten.
        subj_l = _normalise_word(subj)
        verb_l = _normalise_word(verb)
        obj_l = _normalise_word(obj)
        positive = 0
        negative = 0
        for para in self._paragraphs:
            tokens = [tok for s in para.sentences for tok in s.tokens]
            if subj_l in tokens and verb_l in tokens and obj_l in tokens:
                # Bevorzuge eine geordnete Übereinstimmung, sonst als unsicher behandeln.
                if _ordered_contains(tokens, [subj_l, verb_l, obj_l]):
                    positive += 1
                else:
                    negative += 1
            elif subj_l in tokens and verb_l in tokens:
                negative += 1
        confidence = positive / (positive + negative) if (positive + negative) else 0.0
        return positive, negative, confidence

    def targeted_replay(self, chapter_idx: int, paragraph_indices: Sequence[int]) -> None:
        """Mimik eines Replays durch Anpassen der Replay-Temperatur und des Gates."""
        # Diese Logik ist ein Platzhalter für die Steuerung des Transformer-Trainings/Feinabstimmung.
        self.system.replay_temp = max(0.1, self.system.replay_temp * 0.9)
        if paragraph_indices:
            self.system.hard_gate_countdown = max(self.system.hard_gate_countdown - 1, 0)

    # ------------------------------------------------------------------
    # Hooks overridden by ``InstrumentedBookAdapter``
    # ------------------------------------------------------------------
    def _stimulate_sentence(self, enc: EncodedSentence, ticks_after: int = 6) -> None:
        # Die Stimulationslogik wird an Embeddings angepasst.
        self.cortex.observe(enc.embedding)
        familiarity = self._sentence_familiarity(enc.tokens) # Noch Token-basiert
        novelty = 1.0 - _cosine_embedding(enc.embedding, self._last_embedding) if self._last_embedding else 1.0
        mismatch = novelty * (1.0 - 0.5 * familiarity)
        mismatch = max(0.0, min(1.0, mismatch))
        self.system.ca1_last_mismatch = [mismatch]
        self._last_embedding = enc.embedding
        self._token_mastery.update(enc.tokens) # Beibehalten als Platzhalter
        self._update_familiarity_trace(familiarity)

        # Temperatur passt sich basierend auf lexikalischer Neuheit an.
        base = getattr(self.system, "replay_temp", 1.0)
        self.system.replay_temp = max(0.05, min(5.0, base * (1.0 + 0.1 * (mismatch - 0.5))))
        # Hard Gate aktiviert, wenn Neuheit zu hoch ist.
        if mismatch > 0.8:
            self.system.hard_gate_countdown = 3
        elif self.system.hard_gate_countdown > 0:
            self.system.hard_gate_countdown -= 1

    def _sentence_familiarity(self, tokens: Sequence[str]) -> float:
        # Diese Metrik ist noch Token-basiert. Für Transformer könnte sie
        # durch die Analyse von Embeddings oder Modell-Confidence ersetzt werden.
        if not tokens:
            return 0.0
        mastery = sum(self._token_mastery.get(tok, 0) for tok in tokens) / len(tokens)
        return 1.0 - math.exp(-mastery / 3.0)

    def _update_familiarity_trace(self, familiarity: float) -> None:
        baseline = getattr(self.system, "average_familiarity", 0.0)
        updated = baseline * 0.9 + familiarity * 0.1
        self.system.average_familiarity = max(0.0, min(1.0, updated))

    def _engram_saliency(self) -> float:
        # Die Salienz wird jetzt aus Embedding-basierter Kohärenz und Familiarität abgeleitet.
        coherence = self.cortex.coherence_score()
        familiarity = getattr(self.system, "average_familiarity", 0.0)
        combined = 0.5 * coherence + 0.5 * familiarity
        return max(0.05, min(1.0, combined))

    def compress_chapter(self, chapter_idx: int) -> None:
        # Leichtes Absinken der Replay-Temperatur nach einem Kapitel, um Konsolidierung zu imitieren.
        # Dies könnte im Transformer-Kontext eine Phase der Modell-Konsolidierung oder des Fine-Tunings darstellen.
        self.system.replay_temp = max(0.1, self.system.replay_temp * 0.95)

    def monitor_and_intervene(self, chapter_idx: int) -> None:
        # Zufälliges Zurücksetzen des Hard Gates, um Operator-Interventionen zu emulieren.
        # Im Transformer-Kontext könnte dies eine Anpassung der Trainingsstrategie sein.
        if self.system.hard_gate_countdown > 0 and self._rng.random() < 0.3:
            self.system.hard_gate_countdown -= 1

    # Interne Hilfsfunktionen -------------------------------------------------
    def _context_ids_for(self, tokens: Sequence[str]) -> List[int]:
        # Diese Funktion ist für die SNN-Architektur spezifisch und wird für Transformer nicht direkt benötigt.
        # Sie wird beibehalten, um die Kompatibilität mit der EncodedSentence-Struktur zu gewährleisten,
        # aber die generierten IDs haben keine direkte Bedeutung für das Transformer-Modell.
        # Stabile Hash-Funktion, um Pseudo-Kontext-Neuron-IDs abzuleiten.
        ctx_ids = {abs(hash(tok)) % 997 for tok in tokens}
        if not ctx_ids:
            return [0]
        return sorted(ctx_ids)


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def _ordered_contains(tokens: Sequence[str], pattern: Sequence[str]) -> bool:
    if len(pattern) > len(tokens):
        return False
    it = iter(tokens)
    try:
        for target in pattern:
            while True:
                current = next(it)
                if current == target:
                    break
        return True
    except StopIteration:
        return False

from __future__ import annotations
# ------------------------------------------------------------
# Add-on: Checkpointing für Transformer-Modell
# Nur Standardbibliothek, Python 3.12
# ------------------------------------------------------------

import json
import os
import torch
from typing import Dict, Any, Optional, Tuple

# --- Importiere das neue Transformer-Modul
try:
    import transformer_core as tc
except ImportError:
    print("Warnung: 'transformer_core' Modul nicht gefunden. Checkpointing für Transformer nicht verfügbar.")
    tc = None

# ============================================================
# 1) Transformer-Checkpointing
# ============================================================

def save_transformer_model(model: tc.TransformerModel, optimizer: torch.optim.Optimizer,
                           epoch: int, loss: float, path: str) -> None:
    """
    Speichert den Zustand eines Transformer-Modells, Optimierers und Trainingsmetriken.
    """
    if tc is None:
        raise RuntimeError("Transformer-Modul nicht verfügbar. Speichern nicht möglich.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model.config.to_dict() # Speichere die Konfiguration des Modells
    }
    torch.save(checkpoint, path)
    print(f"Transformer-Modell-Checkpoint gespeichert in: {path}")

def load_transformer_model(path: str, device: str = 'cpu') -> Tuple[tc.TransformerModel, torch.optim.Optimizer, int, float]:
    """
    Lädt ein Transformer-Modell, Optimierer und Trainingsmetriken von einem Checkpoint.
    """
    if tc is None:
        raise RuntimeError("Transformer-Modul nicht verfügbar. Laden nicht möglich.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint-Datei nicht gefunden: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Modellkonfiguration laden und Modell instanziieren
    model_config = tc.TransformerConfig.from_dict(checkpoint['model_config'])
    model = tc.TransformerModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Optimierer instanziieren und Zustand laden
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate) # Annahme: AdamW
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Transformer-Modell-Checkpoint geladen von: {path} (Epoche: {epoch}, Verlust: {loss:.4f})")
    return model, optimizer, epoch, loss

def save_transformer_config(config: tc.TransformerConfig, path: str) -> None:
    """
    Speichert die Konfiguration eines Transformer-Modells als JSON.
    """
    if tc is None:
        raise RuntimeError("Transformer-Modul nicht verfügbar. Konfiguration speichern nicht möglich.")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"Transformer-Konfiguration gespeichert in: {path}")

def load_transformer_config(path: str) -> tc.TransformerConfig:
    """
    Lädt die Konfiguration eines Transformer-Modells von einer JSON-Datei.
    """
    if tc is None:
        raise RuntimeError("Transformer-Modul nicht verfügbar. Konfiguration laden nicht möglich.")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    print(f"Transformer-Konfiguration geladen von: {path}")
    return tc.TransformerConfig.from_dict(config_dict)


# ============================================================
# 2) Platzhalter für zukünftige Integration/Migration
# ============================================================

# Die ursprünglichen SNN-spezifischen Funktionen wurden entfernt,
# da der Plan eine Neuausrichtung auf Transformer vorsieht.
# Dieser Teil dient als Schnittstelle für das neue System.

# Beispiel für eine zukünftige Funktion, die SNN-Konzepte in Transformer-Metriken übersetzt
def get_transformer_metrics(model: tc.TransformerModel) -> Dict[str, Any]:
    """
    Sammelt und bereitet Metriken des Transformer-Modells für die Anzeige auf.
    Dies könnte z.B. die Anzahl der Parameter, die aktuelle Lernrate,
    oder spezifische Aufmerksamkeitsmuster (falls interpretierbar) umfassen.
    """
    if tc is None:
        return {"Status": "Transformer-Modul nicht geladen"}

    metrics = {
        "Anzahl Parameter": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "Aktuelle Lernrate": "N/A", # Muss vom Optimierer extrahiert werden
        "Modell-Dimension": model.config.d_model,
        "Anzahl Layer": model.config.num_layers,
        "Anzahl Attention-Heads": model.config.num_heads,
        "Vokabulargröße": model.config.vocab_size,
        "Max. Sequenzlänge": model.config.max_seq_len,
    }
    # Versuch, die Lernrate zu extrahieren (abhängig vom Optimierer)
    if hasattr(model, '_optimizer') and model._optimizer is not None:
        for param_group in model._optimizer.param_groups:
            metrics["Aktuelle Lernrate"] = param_group['lr']
            break
    return metrics

# ------------------------------------------------------------
# Convenience-Funktionen (Public API)
# ------------------------------------------------------------

def save_model_and_config(model: tc.TransformerModel, optimizer: torch.optim.Optimizer,
                          epoch: int, loss: float, model_path: str, config_path: str) -> None:
    """
    Speichert das Transformer-Modell und seine Konfiguration.
    """
    save_transformer_model(model, optimizer, epoch, loss, model_path)
    save_transformer_config(model.config, config_path)

def load_model_and_config(model_path: str, config_path: str, device: str = 'cpu') -> Tuple[tc.TransformerModel, torch.optim.Optimizer, int, float]:
    """
    Lädt das Transformer-Modell und seine Konfiguration.
    """
    # Zuerst die Konfiguration laden, um das Modell korrekt zu initialisieren
    config = load_transformer_config(config_path)
    
    # Dann das Modell und den Optimierer laden
    model, optimizer, epoch, loss = load_transformer_model(model_path, device)
    
    # Sicherstellen, dass die geladene Modellkonfiguration mit der im Checkpoint übereinstimmt
    if model.config.to_dict() != config.to_dict():
        print("Warnung: Geladene Modellkonfiguration aus separater Datei weicht von der im Checkpoint ab.")
    
    return model, optimizer, epoch, loss

# ============================================================
# 3) Mini-Demo (optional)
# ============================================================

def _demo():
    if tc is None:
        print("Demo übersprungen: Transformer-Modul nicht verfügbar.")
        return

    print(">> Starte Transformer-Checkpointing Demo…")

    # 1. Modell und Optimierer initialisieren
    config = tc.TransformerConfig(
        vocab_size=1000, d_model=128, num_layers=2, num_heads=4,
        d_ff=256, max_seq_len=50, dropout=0.1, learning_rate=0.001
    )
    model = tc.TransformerModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Simulierte Trainingsschritte
    epoch = 5
    loss = 0.1234

    # 2. Checkpoint speichern
    model_ckpt_path = "demo_transformer_model.pth"
    config_json_path = "demo_transformer_config.json"
    save_model_and_config(model, optimizer, epoch, loss, model_ckpt_path, config_json_path)

    # 3. Modell laden
    print("\n>> Lade Transformer-Modell von Checkpoint…")
    loaded_model, loaded_optimizer, loaded_epoch, loaded_loss = load_model_and_config(
        model_ckpt_path, config_json_path
    )

    print(f"Geladene Epoche: {loaded_epoch}, Geladener Verlust: {loaded_loss:.4f}")
    print(f"Modell-Konfiguration (geladen): {loaded_model.config.to_dict()}")

    # 4. Metriken abrufen
    print("\n>> Abrufen von Transformer-Metriken…")
    metrics = get_transformer_metrics(loaded_model)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Aufräumen
    os.remove(model_ckpt_path)
    os.remove(config_json_path)
    print("\nDemo abgeschlossen und Dateien entfernt.")

if __name__ == "__main__":
    _demo()

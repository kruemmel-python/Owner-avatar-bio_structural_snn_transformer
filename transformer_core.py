import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class Tokenizer:
    """
    Kapselt transformers.AutoTokenizer für einfache Textkodierung und -dekodierung.
    """
    def __init__(self, pretrained_model_name_or_path: str = "gpt2"):
        """
        Initialisiert den Tokenizer.
        Args:
            pretrained_model_name_or_path: Der Name oder Pfad des vorab trainierten Tokenizers.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        # Füge ein Padding-Token hinzu, falls nicht vorhanden, und setze es als pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Stelle sicher, dass das Modell die neue Vokabulargröße kennt, falls der Tokenizer erweitert wurde.
        # Dies muss im Modell selbst gehandhabt werden, wenn es initialisiert wird.

    def encode(self, text: str, max_length: int = None, truncation: bool = True, padding: str = 'max_length') -> torch.Tensor:
        """
        Kodiert einen Text in Token-IDs.
        Args:
            text: Der zu kodierende Text.
            max_length: Maximale Sequenzlänge.
            truncation: Ob der Text abgeschnitten werden soll, wenn er länger als max_length ist.
            padding: Padding-Strategie ('max_length', 'longest', 'do_not_pad').
        Returns:
            Ein Tensor von Token-IDs.
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=truncation,
            padding=padding
        )
        return encoded.input_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Dekodiert Token-IDs zurück in Text.
        Args:
            token_ids: Eine Liste von Token-IDs.
            skip_special_tokens: Ob spezielle Token (z.B. [PAD], [CLS]) übersprungen werden sollen.
        Returns:
            Der dekodierte Text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        """
        Gibt die Größe des Vokabulars zurück.
        """
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """
        Gibt die Token-ID des Padding-Tokens zurück.
        """
        return self.tokenizer.pad_token_id

    @property
    def name(self) -> str:
        """
        Gibt den Namen des vorab trainierten Tokenizers zurück.
        """
        return self.tokenizer.name_or_path


class PositionalEncoding(nn.Module):
    """
    Fügt Positionsinformationen zu den Eingabe-Embeddings hinzu.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embed_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """
    Implementiert Multi-Head Self-Attention.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model muss durch num_heads teilbar sein"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)

        # 1) Lineare Transformationen und Aufteilung in Köpfe
        query = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Skaliertes Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Maskiere unerwünschte Verbindungen

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        # 3) Konkatenation der Köpfe und finale lineare Transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        return output

class FeedForward(nn.Module):
    """
    Implementiert eine einfache Feed-Forward-Schicht.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class DecoderLayer(nn.Module):
    """
    Ein einzelner Decoder-Layer des Transformers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, look_ahead_mask: torch.Tensor) -> torch.Tensor:
        # Self-Attention mit Maskierung
        attn_output = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout1(attn_output)) # Add & Norm

        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output)) # Add & Norm
        return x

class TransformerCore(nn.Module):
    """
    Der Kern des Transformer LLM.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generiert eine kausale Maske für Self-Attention.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len] (kausale Maske)
        Returns:
            output: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        # src muss [seq_len, batch_size] sein für PositionalEncoding
        src = src.transpose(0, 1) # [seq_len, batch_size]

        if src_mask is None:
            # Erstelle eine kausale Maske, wenn keine bereitgestellt wird
            seq_len = src.size(0)
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
            # Die Maske muss für Multi-Head Attention die Form [1, num_heads, seq_len, seq_len] haben
            src_mask = src_mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)

        for layer in self.decoder_layers:
            src = layer(src, src_mask)

        output = self.output_linear(src) # [seq_len, batch_size, vocab_size]
        output = output.transpose(0, 1) # [batch_size, seq_len, vocab_size]
        return output

class TransformerTrainer:
    """
    Klasse zur Kapselung der Trainingslogik für den Transformer.
    """
    def __init__(self, model: TransformerCore, optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, device: torch.device, tokenizer: Tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tokenizer = tokenizer # Speichere den Tokenizer
        self.model.to(self.device)
        self.data_loader: DataLoader = None # Wird später mit Trainingsdaten gesetzt
        self.learning_rate = optimizer.param_groups[0]['lr'] # Initialisiere Lernrate

    def reset_optimizer(self, learning_rate: float = None):
        """
        Setzt den Optimierer mit der aktuellen oder einer neuen Lernrate zurück.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print(f"Optimierer zurückgesetzt mit Lernrate: {self.learning_rate}")

    def set_learning_rate(self, lr: float):
        """
        Setzt die Lernrate des Optimierers.
        """
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Lernrate auf {lr} gesetzt.")

    def set_training_data(self, data_loader: DataLoader):
        """
        Setzt den DataLoader für die Trainingsdaten.
        """
        self.data_loader = data_loader
        print(f"Trainingsdaten-Loader gesetzt. Anzahl der Batches: {len(self.data_loader) if self.data_loader else 0}")

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """
        Führt einen einzelnen Trainingsschritt durch.
        Args:
            input_ids: Batch von Eingabe-Token-IDs, shape [batch_size, seq_len]
            target_ids: Batch von Ziel-Token-IDs, shape [batch_size, seq_len]
        Returns:
            Verlustwert für diesen Schritt.
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Der Transformer erwartet src als [batch_size, seq_len]
        # Die Maske wird intern generiert, wenn nicht explizit übergeben
        output = self.model(input_ids) # [batch_size, seq_len, vocab_size]

        # Für die Verlustberechnung müssen wir die Dimensionen anpassen
        # output: [batch_size * seq_len, vocab_size]
        # target_ids: [batch_size * seq_len]
        loss = self.criterion(output.view(-1, output.size(-1)), target_ids.view(-1))

        loss.backward()
        # Optional: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self) -> float:
        """
        Führt eine komplette Trainings-Epoche durch.
        Returns:
            Durchschnittlicher Verlust für die Epoche.
        """
        if self.data_loader is None:
            raise ValueError("Trainingsdaten-Loader wurde nicht gesetzt. Bitte rufen Sie 'set_training_data' auf.")

        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(self.data_loader):
            loss = self.train_step(input_ids, target_ids)
            total_loss += loss
            num_batches += 1
            # Optional: Fortschritt ausgeben
            # if batch_idx % 100 == 0:
            #     print(f"  Batch {batch_idx}/{len(self.data_loader)}, Verlust: {loss:.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """
        Führt einen einzelnen Evaluationsschritt durch.
        Args:
            input_ids: Batch von Eingabe-Token-IDs, shape [batch_size, seq_len]
            target_ids: Batch von Ziel-Token-IDs, shape [batch_size, seq_len]
        Returns:
            Verlustwert für diesen Schritt.
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids)
            loss = self.criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
        return loss.item()

    def save_model(self, path: str):
        """Speichert den Modellzustand und den Optimiererzustand."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer_name': self.tokenizer.name, # Speichere den Namen des Tokenizers
            'vocab_size': self.tokenizer.vocab_size, # Speichere auch die Vokabulargröße
            'd_model': self.model.d_model,
            'num_layers': len(self.model.decoder_layers),
            'num_heads': self.model.decoder_layers[0].self_attn.num_heads, # Annahme: alle Layer haben gleiche num_heads
            'd_ff': self.model.decoder_layers[0].feed_forward.linear_1.out_features, # Annahme: alle Layer haben gleiche d_ff
            'max_seq_len': self.model.pos_encoder.pe.size(0),
            'dropout': self.model.dropout.p, # Annahme: alle Dropouts haben den gleichen Wert
            'learning_rate': self.learning_rate # Speichere die aktuelle Lernrate
        }, path)
        # Speichere den Tokenizer separat, da er nicht Teil des PyTorch-Modells ist
        # Der Tokenizer wird über seinen Namen geladen, nicht über einen Pfad
        # self.tokenizer.tokenizer.save_pretrained(path + "_tokenizer") # Nicht mehr nötig, da Name gespeichert wird
        print(f"Modell, Optimierer und Tokenizer-Name in {path} gespeichert.")

    def load_model(self, path: str):
        """Lädt den Modellzustand und den Optimiererzustand."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate) # Lade Lernrate, falls vorhanden
        # Lade den Tokenizer neu basierend auf dem gespeicherten Namen
        tokenizer_name = checkpoint.get('tokenizer_name', 'gpt2') # Standard auf gpt2, falls nicht gespeichert
        self.tokenizer = Tokenizer(pretrained_model_name_or_path=tokenizer_name)
        print(f"Modell, Optimierer und Tokenizer von {path} geladen.")

    def generate_text(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> list[int]:
        """
        Generiert Text basierend auf einem Start-Prompt.
        Args:
            prompt_ids: Tensor der Start-Token-IDs, shape [1, current_seq_len]
            max_new_tokens: Maximale Anzahl der zu generierenden neuen Token.
            temperature: Sampling-Temperatur für die Softmax-Verteilung.
        Returns:
            Liste der generierten Token-IDs.
        """
        self.model.eval()
        generated_ids = prompt_ids.tolist()[0] # Start mit den Prompt-IDs
        current_input_ids = prompt_ids.to(self.device)

        for _ in range(max_new_tokens):
            # Beschränke die Eingabesequenz auf die maximale Sequenzlänge des Modells
            # Dies ist wichtig, da PositionalEncoding eine feste max_len hat
            max_model_seq_len = self.model.pos_encoder.pe.size(0)
            if current_input_ids.size(1) > max_model_seq_len:
                current_input_ids = current_input_ids[:, -max_model_seq_len:]

            with torch.no_grad():
                output = self.model(current_input_ids) # [1, current_seq_len, vocab_size]

            # Nächstes Token vorhersagen (letztes Token der Sequenz)
            logits = output[:, -1, :] # [1, vocab_size]

            if temperature == 0: # Greedy Sampling
                next_token_id = torch.argmax(logits, dim=-1).item()
            else: # Top-k oder Nucleus Sampling könnten hier implementiert werden
                probabilities = F.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()

            generated_ids.append(next_token_id)
            current_input_ids = torch.cat([current_input_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)

        return generated_ids

if __name__ == '__main__':
    # Beispielnutzung und Test
    print("Starte TransformerCore Beispieltest...")

    # Hyperparameter
    d_model = 512
    num_layers = 4
    num_heads = 8
    d_ff = 2048
    max_seq_len = 256
    dropout = 0.1
    batch_size = 2
    seq_len = 128
    learning_rate = 1e-4
    epochs = 3

    # Gerät auswählen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    # Tokenizer initialisieren
    tokenizer = Tokenizer(pretrained_model_name_or_path="gpt2")
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    print(f"Tokenizer initialisiert mit Vokabulargröße: {vocab_size}")
    print(f"Padding Token ID: {pad_token_id}")

    # Modell initialisieren
    model = TransformerCore(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout).to(device)
    print("Modell initialisiert.")
    print(model)

    # Optimizer und Verlustfunktion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id) # Verwende pad_token_id als ignore_index

    # Trainer initialisieren
    trainer = TransformerTrainer(model, optimizer, criterion, device, tokenizer)
    print("Trainer initialisiert.")

    # Simulierte Daten
    print("Simuliere Trainingsdaten...")
    # Erstelle Dummy-Daten, die das Padding-Token enthalten können
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Optional: Setze einige auf pad_token_id
    dummy_input_ids[:, seq_len // 2:] = pad_token_id
    dummy_target_ids[:, seq_len // 2:] = pad_token_id

    # Erstelle einen Dummy DataLoader
    from torch.utils.data import TensorDataset
    dummy_dataset = TensorDataset(dummy_input_ids, dummy_target_ids)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size)
    trainer.set_training_data(dummy_dataloader)


    # Trainingsschleife
    print(f"Starte Training für {epochs} Epochen...")
    for epoch in range(epochs):
        loss = trainer.train_epoch() # Nutze die neue train_epoch Methode
        print(f"Epoche {epoch+1}/{epochs}, Trainingsverlust: {loss:.4f}")

    # Evaluationsschritt
    eval_loss = trainer.evaluate_step(dummy_input_ids, dummy_target_ids)
    print(f"Evaluationsverlust nach Training: {eval_loss:.4f}")

    # Modell speichern und laden
    model_path = "transformer_core_model.pt"
    trainer.save_model(model_path)
    # Erstelle einen neuen Trainer, um das Laden zu testen
    # Der Tokenizer wird jetzt intern im load_model neu erstellt
    new_model = TransformerCore(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout).to(device)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)
    new_trainer = TransformerTrainer(new_model, new_optimizer, criterion, device, Tokenizer("gpt2")) # Temporärer Tokenizer für Initialisierung
    new_trainer.load_model(model_path)
    print("Modell erfolgreich geladen und getestet.")

    # Test reset_optimizer und set_learning_rate
    print("\nTeste Optimierer-Reset und Lernraten-Setzung...")
    old_lr = new_trainer.optimizer.param_groups[0]['lr']
    new_trainer.set_learning_rate(old_lr / 2)
    print(f"Neue Lernrate: {new_trainer.optimizer.param_groups[0]['lr']}")
    new_trainer.reset_optimizer(learning_rate=old_lr * 2)
    print(f"Lernrate nach Reset: {new_trainer.optimizer.param_groups[0]['lr']}")


    # Textgenerierung
    print("\nTeste Textgenerierung...")
    # Ein einfacher Prompt (z.B. "Hello world")
    prompt_text = "Hello world"
    prompt_ids = new_trainer.tokenizer.encode(prompt_text, max_length=max_seq_len, truncation=True, padding='do_not_pad')
    print(f"Prompt Text: '{prompt_text}'")
    print(f"Prompt IDs: {prompt_ids.tolist()}")

    generated_sequence_ids = new_trainer.generate_text(prompt_ids, max_new_tokens=20, temperature=0.7)
    generated_text = new_trainer.tokenizer.decode(generated_sequence_ids, skip_special_tokens=True)
    print(f"Generierte Sequenz (Token-IDs): {generated_sequence_ids}")
    print(f"Generierter Text: '{generated_text}'")

    print("TransformerCore Test abgeschlossen.")

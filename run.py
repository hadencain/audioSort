import filecmp
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import mutagen
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Maps YAMNet AudioSet class display names → (category, subcategory or None)
YAMNET_CLASS_MAP = {
    "Drum": ("drums", "loop"),
    "Snare drum": ("drums", "snare"),
    "Rimshot": ("drums", "snare"),
    "Bass drum": ("drums", "kick"),
    "Drum roll": ("drums", "loop"),
    "Hi-hat": ("drums", "hihat"),
    "Cymbal": ("drums", "crash"),
    "Tambourine": ("drums", "shakers"),
    "Maraca": ("drums", "shakers"),
    "Cowbell": ("drums", "percussion"),
    "Tabla": ("drums", "percussion"),
    "Conga drum": ("drums", "percussion"),
    "Bongo drum": ("drums", "percussion"),
    "Clapping": ("drums", "percussion"),
    "Piano": ("keys", None),
    "Electric piano": ("keys", None),
    "Keyboard (musical)": ("keys", None),
    "Organ": ("keys", None),
    "Hammond organ": ("keys", None),
    "Synthesizer": ("synth", None),
    "Electric guitar": ("strings", None),
    "Acoustic guitar": ("strings", None),
    "Guitar": ("strings", None),
    "Bass guitar": ("bass", None),
    "Violin, fiddle": ("strings", None),
    "Cello": ("strings", None),
    "Banjo": ("strings", None),
    "Mandolin": ("strings", None),
    "Ukulele": ("strings", None),
    "Harp": ("strings", None),
    "Trumpet": ("brass", None),
    "Trombone": ("brass", None),
    "French horn": ("brass", None),
    "Saxophone": ("brass", None),
    "Brass instrument": ("brass", None),
    "Flute": ("woodwinds", None),
    "Clarinet": ("woodwinds", None),
    "Oboe": ("woodwinds", None),
    "Woodwind instrument": ("woodwinds", None),
    "Bell": ("bells", None),
    "Glockenspiel": ("bells", None),
    "Marimba, xylophone": ("bells", None),
    "Vibraphone, vibraharp": ("bells", None),
    "Singing": ("vocals", None),
    "Male singing": ("vocals", None),
    "Female singing": ("vocals", None),
    "Choir": ("vocals", None),
    "Chant": ("vocals", None),
    "Rapping": ("vocals", None),
    "Speech": ("vocals", None),
    "Ambient music": ("pad", None),
    "Drone": ("pad", None),
    "Noise": ("fx", None),
    "Whoosh, swoosh, swish": ("fx", None),
    "Reverberation": ("fx", None),
    "Distortion": ("fx", None),
}

DEFAULT_CONFIG = {
    "use_path_context": True,
    "use_metadata": True,
    "use_audio_features": True,
    "use_ml_classifier": True,
    "ml_model": "yamnet",
    "min_confidence_threshold": 0.15,
    "signal_weights": {
        "path": 2.5,
        "metadata": 2.0,
        "filename": 1.0,
        "audio": 1.5,
        "ml": 2.0,
    },
}


AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


class SignalScorer:
    def __init__(self, categories, config):
        self.categories = categories
        self.config = config
        self.weights = config.get("signal_weights", DEFAULT_CONFIG["signal_weights"])
        self._ast_model = None
        self._ast_extractor = None
        self._ast_id2label = None
        if config.get("use_ml_classifier", True) and HAS_TORCH:
            self._load_ast()

    def _load_ast(self):
        try:
            logging.info("Loading AST model (first run downloads ~330MB)...")
            self._ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL)
            self._ast_model = ASTForAudioClassification.from_pretrained(AST_MODEL)
            self._ast_model.eval()
            self._ast_id2label = self._ast_model.config.id2label
            logging.info("AST model ready")
        except Exception as e:
            logging.warning(f"AST model load failed, skipping ML signal: {e}")
            self._ast_model = None

    def score(self, file_path: Path):
        """Returns (scores_dict, subcategory_votes_dict)."""
        scores = defaultdict(float)
        sub_votes = defaultdict(lambda: defaultdict(float))

        if self.config.get("use_path_context", True):
            self._score_path(file_path, scores, sub_votes)
        if self.config.get("use_metadata", True) and HAS_MUTAGEN:
            self._score_metadata(file_path, scores, sub_votes)
        self._score_filename(file_path, scores, sub_votes)
        if self.config.get("use_audio_features", True) and HAS_LIBROSA:
            self._score_audio_features(file_path, scores, sub_votes)
        if self.config.get("use_ml_classifier", True) and self._ast_model and HAS_LIBROSA:
            self._score_ml(file_path, scores, sub_votes)

        return scores, sub_votes

    # --- keyword helpers ---

    def _apply_keywords(self, text, scores, sub_votes, weight):
        text_lower = text.lower()
        for category, keywords in self.categories.items():
            if isinstance(keywords, dict):
                for subcategory, sub_keywords in keywords.items():
                    hits = sum(1 for kw in sub_keywords if kw.lower() in text_lower)
                    if hits:
                        scores[category] += weight * hits
                        sub_votes[category][subcategory] += weight * hits
            else:
                hits = sum(1 for kw in keywords if kw.lower() in text_lower)
                if hits:
                    scores[category] += weight * hits

    # --- signal methods ---

    def _score_path(self, file_path, scores, sub_votes):
        w = self.weights.get("path", 2.5)
        for part in file_path.parts[:-1]:  # all folder names, not the filename
            self._apply_keywords(part, scores, sub_votes, w)

    def _score_metadata(self, file_path, scores, sub_votes):
        w = self.weights.get("metadata", 2.0)
        try:
            audio = mutagen.File(file_path)
            if audio is None:
                return
            texts = []
            for key in ("TIT2", "COMM", "TXXX", "TCON"):
                if key in audio:
                    texts.append(str(audio[key]))
            tags = getattr(audio, "tags", None)
            if tags:
                for key in ("genre", "comment", "description", "category", "title"):
                    if key in tags:
                        texts.append(str(tags[key]))
            for text in texts:
                self._apply_keywords(text, scores, sub_votes, w)
        except Exception:
            pass

    def _score_filename(self, file_path, scores, sub_votes):
        w = self.weights.get("filename", 1.0)
        self._apply_keywords(file_path.stem, scores, sub_votes, w)

    def _score_audio_features(self, file_path, scores, sub_votes):
        w = self.weights.get("audio", 1.5)
        try:
            y, sr = librosa.load(str(file_path), sr=22050, mono=True, duration=10.0)
            if len(y) == 0:
                return

            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            duration = len(y) / sr
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_count = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))

            # Very low centroid → kick/sub
            if centroid < 300:
                scores["drums"] += w * 1.5
                sub_votes["drums"]["kick"] += w * 1.5
                scores["bass"] += w * 0.8

            # Low-mid centroid → bass or keys
            elif centroid < 800:
                scores["bass"] += w * 0.7
                scores["keys"] += w * 0.5

            # High centroid + noisy → hi-hat / cymbal
            if centroid > 4000 and zcr > 0.2:
                scores["drums"] += w * 1.2
                sub_votes["drums"]["hihat"] += w * 1.2

            # High ZCR generally → percussive content
            if zcr > 0.15:
                scores["drums"] += w * 0.4

            # Long, sparse onsets → pad / ambient
            if duration > 2.0 and onset_count < 4:
                scores["pad"] += w * 1.0

            # Dense onsets → drum loop
            if onset_count > 8 and duration > 0.5:
                scores["drums"] += w * 0.8
                sub_votes["drums"]["loop"] += w * 0.8

            # Single short transient → one-shot kick or snare
            if onset_count <= 2 and duration < 0.8:
                if centroid < 400:
                    scores["drums"] += w * 1.0
                    sub_votes["drums"]["kick"] += w * 1.0
                elif centroid < 2000 and zcr > 0.1:
                    scores["drums"] += w * 0.8
                    sub_votes["drums"]["snare"] += w * 0.8

        except Exception as e:
            logging.debug(f"Audio features failed for {file_path.name}: {e}")

    def _score_ml(self, file_path, scores, sub_votes):
        w = self.weights.get("ml", 2.0)
        try:
            y, sr = librosa.load(str(file_path), sr=16000, mono=True, duration=10.0)
            if len(y) == 0:
                return
            inputs = self._ast_extractor(
                y.tolist(), sampling_rate=16000, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self._ast_model(**inputs).logits
            class_scores = torch.sigmoid(logits)[0].numpy()

            for idx, class_name in self._ast_id2label.items():
                if class_name in YAMNET_CLASS_MAP and class_scores[idx] > 0.05:
                    category, subcategory = YAMNET_CLASS_MAP[class_name]
                    contribution = float(class_scores[idx]) * w
                    scores[category] += contribution
                    if subcategory:
                        sub_votes[category][subcategory] += contribution

        except Exception as e:
            logging.debug(f"ML scoring failed for {file_path.name}: {e}")


class AudioFileOrganizer:
    def __init__(self, source_dir, destination_dir, config=None):
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self.config = config or DEFAULT_CONFIG
        self.setup_logging()
        self.load_categories()
        self.scorer = SignalScorer(self.categories, self.config)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def load_categories(self):
        # resolve relative to script location first, then CWD
        candidates = [
            Path(__file__).parent / "categories.json",
            Path("categories.json"),
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    self.categories = json.load(f)
                return
        raise FileNotFoundError("categories.json not found")

    def is_duplicate(self, source_path, dest_path, chunk_size=8192):
        if not dest_path.exists():
            return False
        if source_path.stat().st_size != dest_path.stat().st_size:
            return False
        with open(source_path, "rb") as sf, open(dest_path, "rb") as df:
            while True:
                sc, dc = sf.read(chunk_size), df.read(chunk_size)
                if sc != dc:
                    return False
                if not sc:
                    return True

    def find_unique_filename(self, dest_path):
        if not dest_path.exists():
            return dest_path
        base, ext = dest_path.stem, dest_path.suffix
        counter = 1
        while True:
            new_path = dest_path.parent / f"{base}_{counter}{ext}"
            if not new_path.exists():
                return new_path
            counter += 1

    def create_directory_structure(self):
        for category, keywords in self.categories.items():
            (self.destination_dir / category).mkdir(parents=True, exist_ok=True)
            if isinstance(keywords, dict):
                for subcategory in keywords:
                    (self.destination_dir / category / subcategory).mkdir(parents=True, exist_ok=True)
        (self.destination_dir / "others").mkdir(parents=True, exist_ok=True)

    def identify_destination(self, file_path: Path):
        scores, sub_votes = self.scorer.score(file_path)
        threshold = self.config.get("min_confidence_threshold", 0.15)

        if not scores or max(scores.values()) < threshold:
            return "others", None

        best_cat = max(scores, key=scores.get)
        best_sub = None
        if isinstance(self.categories.get(best_cat), dict) and sub_votes[best_cat]:
            best_sub = max(sub_votes[best_cat], key=sub_votes[best_cat].get)

        return best_cat, best_sub

    def organize_files(self):
        supported = {".wav", ".mp3", ".aif", ".aiff", ".flac", ".ogg"}
        log = {"total_files": 0, "copied": 0, "skipped_duplicates": 0, "errors": 0}

        try:
            self.create_directory_structure()
            total = sum(1 for f in self.source_dir.rglob("*") if f.suffix.lower() in supported)
            log["total_files"] = total

            for file_path in self.source_dir.rglob("*"):
                if file_path.suffix.lower() not in supported:
                    continue
                try:
                    category, subcategory = self.identify_destination(file_path)

                    if subcategory:
                        dest = self.destination_dir / category / subcategory / file_path.name
                    else:
                        dest = self.destination_dir / category / file_path.name

                    dest.parent.mkdir(parents=True, exist_ok=True)

                    if self.is_duplicate(file_path, dest):
                        logging.info(f"Duplicate skipped: {file_path.name}")
                        log["skipped_duplicates"] += 1
                        continue

                    if dest.exists():
                        dest = self.find_unique_filename(dest)

                    shutil.copy2(file_path, dest)
                    log["copied"] += 1

                    if total > 0:
                        pct = (log["copied"] + log["skipped_duplicates"]) / total * 100
                        logging.info(f"{pct:.1f}% — {file_path.name} → {dest.relative_to(self.destination_dir)}")

                except Exception as e:
                    logging.error(f"Error processing {file_path.name}: {e}")
                    log["errors"] += 1

            logging.info(
                f"\nDone — copied: {log['copied']}  skipped: {log['skipped_duplicates']}  errors: {log['errors']}"
            )

        except Exception as e:
            logging.error(f"Fatal error: {e}")
            log["errors"] += 1

        return log


def load_config():
    candidates = [Path(__file__).parent / "config.json", Path("config.json")]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                user = json.load(f)
            return {**DEFAULT_CONFIG, **user}
    return DEFAULT_CONFIG


def main():
    config = load_config()
    source_directory = r"C:\Users\haden\Documents\Splice"
    destination_directory = r"C:\Users\haden\Documents\Splice_Sorted"

    organizer = AudioFileOrganizer(source_directory, destination_directory, config)
    log = organizer.organize_files()

    print(
        f"\nTotal: {log['total_files']}  "
        f"Copied: {log['copied']}  "
        f"Skipped: {log['skipped_duplicates']}  "
        f"Errors: {log['errors']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")

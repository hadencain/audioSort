# audioSort

Audio sample organizer. Started as keyword-based filename matching; evolved into a weighted multi-signal classifier — folder path context, file metadata, librosa spectral/onset analysis, and an AST ML model (HuggingFace). Sorts WAV, MP3, AIF, AIFF, FLAC, OGG into categorized folders. All ML deps optional.

## How it works

Each file is scored across up to five signals, each weighted independently:

| Signal | Weight | Requires |
|--------|--------|----------|
| Folder path context | 2.5 | — |
| File metadata tags | 2.0 | `mutagen` |
| Filename keywords | 1.0 | — |
| Spectral/onset features | 1.5 | `librosa` |
| AST ML classifier | 2.0 | `torch` + `transformers` |

The highest-scoring category wins. For drums, subcategory is determined by the same scoring (kick, snare, hihat, tom, crash, ride, shakers, percussion, loop).

The ML signal uses [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (~330MB, downloaded on first run).

## Categories

```
drums/
  kick  snare  hihat  tom  crash  ride  shakers  percussion  loop
strings/
synth/
keys/
pad/
bass/
vocals/
brass/
woodwinds/
bells/
fx/
foley/
others/      ← unmatched files land here
```

## Setup

```bash
pip install -r requirements.txt
```

Deps are tiered — the script runs with just `mutagen` installed. Each additional dep activates another signal layer:

```
mutagen          → metadata tags
librosa + numpy  → spectral/onset features
torch + transformers → AST ML classification
```

First run with ML enabled downloads ~330MB model weights via HuggingFace.

## Usage

Edit the source and destination paths in `run.py`:

```python
source_directory = r"C:\path\to\your\samples"
destination_directory = r"C:\path\to\sorted_output"
```

Then run:

```bash
python run.py
```

Files are copied (not moved). Duplicates are skipped via binary comparison. Name collisions get a counter suffix (`kick_1.wav`, `kick_2.wav`, ...).

## Configuration

`config.json` overrides defaults. Example — disable ML, lower confidence threshold:

```json
{
  "use_ml_classifier": false,
  "min_confidence_threshold": 0.10,
  "signal_weights": {
    "path": 3.0,
    "filename": 1.5
  }
}
```

All keys are optional; unset keys fall back to defaults.

## Extending categories

Edit `categories.json`. Flat list = simple category. Object with sub-keys = category with subcategories (like `drums`). Keywords are case-insensitive at match time.

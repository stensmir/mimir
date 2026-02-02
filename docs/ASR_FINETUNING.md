# Fine-tuning ASR моделей для русского языка

Руководство по дообучению моделей Parakeet и Whisper на русскоязычных данных.

## Оглавление

1. [Обзор](#обзор)
2. [Требования](#требования)
3. [Датасеты](#датасеты)
4. [Подготовка окружения](#подготовка-окружения)
5. [Подготовка данных](#подготовка-данных)
6. [Fine-tuning Parakeet](#fine-tuning-parakeet)
7. [Fine-tuning Whisper](#fine-tuning-whisper)
8. [Экспорт модели](#экспорт-модели)
9. [Интеграция с Mimir](#интеграция-с-mimir)
10. [Troubleshooting](#troubleshooting)

---

## Обзор

### Архитектура

```
┌─────────────────────────────────────┐
│         Windows + RTX 4090          │
│  ┌─────────────────────────────┐    │
│  │  Fine-tuning на русском     │    │
│  │  датасете (Golos/CV)        │    │
│  └──────────────┬──────────────┘    │
│                 ▼                   │
│  ┌─────────────────────────────┐    │
│  │  Экспорт в ONNX             │    │
│  └──────────────┬──────────────┘    │
└─────────────────┼───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│            Mac (Mimir)              │
│  ┌─────────────────────────────┐    │
│  │  Инференс через whisper-rs  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### Выбор модели

| Модель | Плюсы | Минусы |
|--------|-------|--------|
| **Parakeet** | Высокое качество, поддержка NeMo | Требует Python runtime |
| **Whisper** | whisper-rs (нативный Rust), простота | Менее гибкий fine-tuning |

**Рекомендация:** Whisper для простоты интеграции с Mimir (whisper-rs).

---

## Требования

### Аппаратные требования (Windows)

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| Диск | 200GB SSD | 500GB NVMe |
| CUDA | 11.8+ | 12.1+ |

### RTX 4090 возможности

- Batch size: 16-32
- Модель Whisper Large: влезает полностью
- Время обучения: ~2-4 часа на 100 часов данных

---

## Датасеты

### Рекомендуемые датасеты

#### 1. Golos (Sberbank) — Рекомендуется

```
Размер: 1200+ часов
Качество: Высокое
Разнообразие: Много голосов, акценты
Лицензия: Apache 2.0
```

**Скачивание:**
```bash
# Через Hugging Face
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('SberDevices/Golos', 'farfield')
ds.save_to_disk('./golos_dataset')
"
```

Или напрямую:
```bash
# farfield subset (~100 часов, чистые данные)
wget https://sc.link/golos_farfield.tar.gz
tar -xzf golos_farfield.tar.gz
```

#### 2. Common Voice (Mozilla)

```
Размер: 200+ часов (русский)
Качество: Среднее (краудсорсинг)
Лицензия: CC-0
```

**Скачивание:**
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('mozilla-foundation/common_voice_16_1', 'ru', split='train')
ds.save_to_disk('./common_voice_ru')
"
```

#### 3. OpenSTT (Sberbank)

```
Размер: 20,000+ часов
Качество: Разное (YouTube)
Лицензия: CC BY-NC-SA
```

**Примечание:** Огромный размер, рекомендуется использовать subset.

### Структура манифеста

NeMo/Parakeet формат:
```json
{"audio_filepath": "audio/001.wav", "text": "привет как дела", "duration": 2.3}
{"audio_filepath": "audio/002.wav", "text": "сегодня хорошая погода", "duration": 1.8}
```

Whisper/HuggingFace формат:
```json
{
  "audio": {"path": "audio/001.wav", "sampling_rate": 16000},
  "sentence": "привет как дела"
}
```

---

## Подготовка окружения

### Windows + CUDA

```powershell
# 1. Установка CUDA Toolkit 12.1
# Скачать с https://developer.nvidia.com/cuda-downloads

# 2. Проверка CUDA
nvidia-smi

# 3. Создание виртуального окружения
python -m venv .venv-finetune
.\.venv-finetune\Scripts\activate

# 4. Установка PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Проверка GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Для Whisper fine-tuning

```bash
pip install transformers datasets accelerate evaluate jiwer
pip install soundfile librosa
```

### Для Parakeet fine-tuning

```bash
pip install nemo_toolkit[asr]
pip install pytorch-lightning
```

---

## Подготовка данных

### Скрипт конвертации аудио

Создать `scripts/prepare_audio.py`:

```python
#!/usr/bin/env python3
"""Подготовка аудио данных для fine-tuning ASR."""

import os
import json
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse


def convert_audio(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Конвертация аудио в 16kHz mono WAV."""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', str(sample_rate),
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_path
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def get_duration(audio_path: str) -> float:
    """Получение длительности аудио."""
    result = subprocess.run([
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        audio_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


def process_golos(input_dir: str, output_dir: str) -> list:
    """Обработка датасета Golos."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_dir = Path(output_dir) / 'audio'
    audio_dir.mkdir(exist_ok=True)

    manifest = []

    for tsv_file in Path(input_dir).glob('**/*.tsv'):
        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    audio_path, text = parts[0], parts[1]

                    # Конвертация
                    input_audio = Path(input_dir) / audio_path
                    output_audio = audio_dir / f"{Path(audio_path).stem}.wav"

                    if convert_audio(str(input_audio), str(output_audio)):
                        duration = get_duration(str(output_audio))
                        manifest.append({
                            'audio_filepath': str(output_audio),
                            'text': text.lower().strip(),
                            'duration': duration
                        })

    return manifest


def process_common_voice(dataset, output_dir: str) -> list:
    """Обработка датасета Common Voice."""
    from datasets import Audio

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_dir = Path(output_dir) / 'audio'
    audio_dir.mkdir(exist_ok=True)

    # Ресемплинг до 16kHz
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    manifest = []

    for i, item in enumerate(dataset):
        audio = item['audio']
        text = item['sentence']

        output_path = audio_dir / f"cv_{i:06d}.wav"

        # Сохранение WAV
        import soundfile as sf
        sf.write(str(output_path), audio['array'], audio['sampling_rate'])

        duration = len(audio['array']) / audio['sampling_rate']

        manifest.append({
            'audio_filepath': str(output_path),
            'text': text.lower().strip(),
            'duration': duration
        })

    return manifest


def split_manifest(manifest: list, train_ratio: float = 0.95) -> tuple:
    """Разделение на train/val."""
    import random
    random.shuffle(manifest)

    split_idx = int(len(manifest) * train_ratio)
    return manifest[:split_idx], manifest[split_idx:]


def save_manifest(manifest: list, output_path: str):
    """Сохранение манифеста в JSONL формате."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--dataset', choices=['golos', 'common_voice'], required=True)
    args = parser.parse_args()

    if args.dataset == 'golos':
        manifest = process_golos(args.input, args.output)
    else:
        from datasets import load_dataset
        ds = load_dataset('mozilla-foundation/common_voice_16_1', 'ru', split='train')
        manifest = process_common_voice(ds, args.output)

    train, val = split_manifest(manifest)

    save_manifest(train, os.path.join(args.output, 'train_manifest.jsonl'))
    save_manifest(val, os.path.join(args.output, 'val_manifest.jsonl'))

    print(f"Train samples: {len(train)}")
    print(f"Val samples: {len(val)}")
```

### Запуск подготовки

```bash
# Для Golos
python scripts/prepare_audio.py \
    --input ./golos_raw \
    --output ./golos_prepared \
    --dataset golos

# Для Common Voice
python scripts/prepare_audio.py \
    --input ./cv_raw \
    --output ./cv_prepared \
    --dataset common_voice
```

---

## Fine-tuning Whisper

### Скрипт обучения

Создать `scripts/finetune_whisper.py`:

```python
#!/usr/bin/env python3
"""Fine-tuning Whisper на русском языке."""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio, Dataset
import evaluate
import json


# Конфигурация
MODEL_NAME = "openai/whisper-large-v3"  # или whisper-medium для экономии VRAM
OUTPUT_DIR = "./whisper-russian-finetuned"
TRAIN_MANIFEST = "./golos_prepared/train_manifest.jsonl"
VAL_MANIFEST = "./golos_prepared/val_manifest.jsonl"


def load_manifest(path: str) -> Dataset:
    """Загрузка манифеста в Dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'audio': item['audio_filepath'],
                'sentence': item['text']
            })

    ds = Dataset.from_list(data)
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    return ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    # Загрузка модели и процессора
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="russian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Принудительно русский язык
    model.generation_config.language = "russian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Загрузка данных
    train_dataset = load_manifest(TRAIN_MANIFEST)
    val_dataset = load_manifest(VAL_MANIFEST)

    # Препроцессинг
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Метрика WER
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Аргументы обучения
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,  # RTX 4090: можно до 24
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,  # ~50 эпох на 100 часов данных
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Обучение
    trainer.train()

    # Сохранение
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

### Запуск обучения

```bash
# Активация окружения
.\.venv-finetune\Scripts\activate

# Запуск
python scripts/finetune_whisper.py

# Мониторинг (в отдельном терминале)
tensorboard --logdir ./whisper-russian-finetuned/runs
```

### Параметры для RTX 4090

| Параметр | Whisper Large | Whisper Medium |
|----------|--------------|----------------|
| batch_size | 16-24 | 32-48 |
| gradient_checkpointing | True | Optional |
| fp16 | True | True |
| VRAM usage | ~20GB | ~12GB |

---

## Fine-tuning Parakeet

### Скрипт обучения

Создать `scripts/finetune_parakeet.py`:

```python
#!/usr/bin/env python3
"""Fine-tuning Parakeet на русском языке."""

import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager


# Конфигурация
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
OUTPUT_DIR = "./parakeet-russian-finetuned"
TRAIN_MANIFEST = "./golos_prepared/train_manifest.jsonl"
VAL_MANIFEST = "./golos_prepared/val_manifest.jsonl"


def main():
    # Загрузка базовой модели
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)

    # Конфигурация данных
    train_config = {
        "manifest_filepath": TRAIN_MANIFEST,
        "sample_rate": 16000,
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
    }

    val_config = {
        "manifest_filepath": VAL_MANIFEST,
        "sample_rate": 16000,
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": True,
    }

    model.setup_training_data(train_data_config=train_config)
    model.setup_validation_data(val_data_config=val_config)

    # Оптимизатор
    model.setup_optimization(
        optim_config=OmegaConf.create({
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 500,
                "min_lr": 1e-6,
            }
        })
    )

    # Trainer
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=50,
        precision="16-mixed",
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        logger=True,
        log_every_n_steps=10,
        val_check_interval=0.25,
    )

    # Experiment manager
    exp_manager(
        trainer,
        {
            "exp_dir": OUTPUT_DIR,
            "name": "parakeet_russian",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": True,
            "checkpoint_callback_params": {
                "monitor": "val_wer",
                "mode": "min",
                "save_top_k": 3,
            }
        }
    )

    # Обучение
    trainer.fit(model)

    # Сохранение
    model.save_to(f"{OUTPUT_DIR}/parakeet_russian.nemo")
    print(f"Model saved to {OUTPUT_DIR}/parakeet_russian.nemo")


if __name__ == "__main__":
    main()
```

---

## Экспорт модели

### Whisper → ONNX

```python
#!/usr/bin/env python3
"""Экспорт Whisper в ONNX для whisper-rs."""

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

MODEL_PATH = "./whisper-russian-finetuned"
OUTPUT_PATH = "./whisper-russian.onnx"

# Загрузка
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)

# Экспорт encoder
dummy_input = torch.randn(1, 80, 3000)  # mel spectrogram

torch.onnx.export(
    model.model.encoder,
    dummy_input,
    f"{OUTPUT_PATH}/encoder.onnx",
    input_names=["input_features"],
    output_names=["encoder_output"],
    dynamic_axes={
        "input_features": {0: "batch", 2: "time"},
        "encoder_output": {0: "batch", 1: "time"},
    },
    opset_version=14,
)

print(f"Encoder exported to {OUTPUT_PATH}/encoder.onnx")

# Для полного экспорта используйте whisper.cpp convert скрипт
```

### Альтернатива: Конвертация через whisper.cpp

```bash
# Клонирование whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp

# Конвертация HuggingFace → ggml
python models/convert-h5-to-ggml.py \
    ../whisper-russian-finetuned \
    ../whisper-russian-finetuned \
    ./whisper-russian-ggml.bin
```

### Parakeet → ONNX

```python
#!/usr/bin/env python3
"""Экспорт Parakeet в ONNX."""

import nemo.collections.asr as nemo_asr

MODEL_PATH = "./parakeet-russian-finetuned/parakeet_russian.nemo"
OUTPUT_PATH = "./parakeet-russian.onnx"

model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
model.export(OUTPUT_PATH)

print(f"Model exported to {OUTPUT_PATH}")
```

---

## Интеграция с Mimir

### Для Whisper (whisper-rs)

1. Скопировать `whisper-russian-ggml.bin` на Mac
2. Обновить путь в конфигурации Mimir:

```rust
// src-tauri/src/transcription_service.rs
const CUSTOM_MODEL_PATH: &str = "~/Library/Application Support/mimir/models/whisper-russian-ggml.bin";
```

### Для Parakeet

1. Скопировать `parakeet-russian.nemo` в `.venv-parakeet`
2. Обновить `scripts/parakeet_server.py`:

```python
# Использование кастомной модели
MODEL_PATH = os.path.expanduser("~/Library/Application Support/mimir/models/parakeet_russian.nemo")
model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Уменьшить batch_size
per_device_train_batch_size=8

# Включить gradient checkpointing
gradient_checkpointing=True

# Использовать меньшую модель
MODEL_NAME = "openai/whisper-medium"
```

### Медленное обучение

```bash
# Увеличить num_workers
dataloader_num_workers=8

# Использовать pin_memory
pin_memory=True

# Проверить, что CUDA используется
python -c "import torch; print(torch.cuda.is_available())"
```

### Плохой WER после обучения

1. Проверить качество данных (шум, неправильные транскрипции)
2. Уменьшить learning_rate: `1e-5` → `5e-6`
3. Увеличить warmup_steps
4. Добавить больше данных

### Переобучение (overfitting)

- WER на train падает, на val растёт
- Решения:
  - Ранняя остановка (early stopping)
  - Больше данных
  - Dropout / augmentation

---

## Чеклист

- [ ] Установить CUDA 12.1 на Windows
- [ ] Создать виртуальное окружение
- [ ] Скачать датасет (Golos рекомендуется)
- [ ] Подготовить аудио (16kHz mono WAV)
- [ ] Создать манифесты train/val
- [ ] Запустить fine-tuning
- [ ] Мониторить через TensorBoard
- [ ] Экспортировать в ONNX/ggml
- [ ] Протестировать на Mac
- [ ] Интегрировать в Mimir

---

## Ссылки

- [Golos Dataset](https://github.com/salute-developers/golos)
- [Common Voice](https://commonvoice.mozilla.org/ru)
- [Whisper Fine-tuning Guide](https://huggingface.co/blog/fine-tune-whisper)
- [NeMo ASR Tutorial](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [whisper-rs](https://github.com/tazz4843/whisper-rs)

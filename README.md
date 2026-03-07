# Culturally-Adaptive-Emotion-Recognition-For-Indian-Faces

> A CDAC research project (Group 14) that builds a culturally grounded Indian emotion recognition framework using the **Navarasa** system and Vision-Language Models.

---

## Motivation

Most existing emotion recognition models are trained on Western-centric datasets and assume emotions are universal — ignoring cultural variations. In the Indian context, emotions are deeply rooted in the **Navarasa** system, which differs fundamentally from Western emotion theories. Current VLMs consistently misclassify Indian facial expressions because no culturally grounded image–text dataset existed for this domain.

---

## Research Objectives

1. Create a Navarasa-based Indian emotion image–text dataset
2. Generate culturally rich emotion descriptions using LLMs (GPT)
3. Fine-tune Vision-Language Models for Indian emotion recognition
4. Evaluate and compare culturally adapted models against existing VLM baselines

---

## The Navarasa System

| Navarasa | Western Equivalent |
|----------|--------------------|
| Shringara | Love, Beauty |
| Hasya | Joy, Humor |
| Karuna | Compassion, Sadness |
| Raudra | Anger, Fury |
| Veera | Courage, Valor |
| Bhayanaka | Fear, Terror |
| Bibhatsa | Disgust, Aversion |
| Adbhuta | Wonder, Amazement |
| Shanta | Peace, Tranquility |

---

## Dataset Creation

Three existing Indian emotion datasets were merged and re-annotated:

**Pipeline:**
```
Merge All Datasets
      ↓
Separate Images by Emotion
      ↓
Emotion Annotation / Navarasa Mapping
      ↓
Generate Navarasa-based Descriptions (GPT)
      ↓
Create Metadata CSV [Image Path | Emotion | GPT Description]
      ↓
Validate Dataset Consistency
      ↓
Upload to HuggingFace Hub
```

**Final dataset structure per sample:** `image`, `navarasa` (emotion label), `text` (GPT-generated cultural description)

### Dataset Statistics

| Emotion | Count | Share |
|---------|-------|-------|
| Hasya | 5,288 | 34.87% |
| Bibhatsa | 2,985 | 19.68% |
| Shantha | 1,201 | 7.92% |
| Shringara | 1,083 | 7.14% |
| Adbhuta | 1,239 | — |
| Bhayanaka | 922 | 6.08% |
| Raudra | 899 | 5.93% |
| Veera | 808 | 5.33% |
| Karuna | 739 | 4.87% |

The dataset was heavily imbalanced (Hasya dominated at ~35%), motivating the creation of a balanced variant.

**HuggingFace datasets:**

| Dataset | ID |
|---------|----|
| Original (unbalanced) | `Navarasa-9/navarasa_9` |
| Balanced | `Navarasa-9/Navarasa_Balanced` / `Raja6922/navarasa_Balanced` |

---

## Project Structure

```
├── DataAnalysis.ipynb                       # EDA — class distribution, image counts, resolution stats
├── image_to_caption.ipynb                   # BLIP-2 captioning pipeline (image → raw text)
│
├── CLIP_finetune.ipynb                      # CLIP fine-tuning on navarasa_9 (unbalanced)
├── Balanced__clip.ipynb                     # CLIP fine-tuning on Navarasa_Balanced
├── Siglip_finetuning.ipynb                  # SigLIP fine-tuning on navarasa_9
├── Balanced_siglip.ipynb                    # SigLIP fine-tuning on Navarasa_Balanced
│
├── Llama3_2__11B__Vision.ipynb              # Llama 3.2 11B Vision LoRA fine-tuning
├── Navarasa_Qwen2_5_VL__7B__Vision.ipynb   # Qwen2.5-VL 7B LoRA fine-tuning
├── Qwen3_VL__8B__Vision__1_.ipynb          # Qwen3-VL 8B LoRA fine-tuning
└── llava_vicuna13B.ipynb                    # LLaVA-v1.6-Vicuna-13B zero-shot inference
```

---

## Models & Approach

### Contrastive Models (Image–Text Retrieval)

Fine-tuned using contrastive loss on Navarasa image–text pairs.

| Model | Base | Dataset |
|-------|------|---------|
| CLIP | `openai/clip-vit-base-patch32` | Unbalanced + Balanced |
| SigLIP | `google/siglip-*` | Balanced |

### Large Vision-Language Models (LoRA Fine-tuning)

Instruction-tuned for direct Navarasa classification using [Unsloth](https://github.com/unslothai/unsloth) + PEFT.

| Model | Parameters | Quantization |
|-------|-----------|--------------|
| Llama 3.2 Vision Instruct | 11B | 4-bit LoRA |
| Qwen2.5-VL Instruct | 7B | 4-bit LoRA |
| Qwen3-VL Instruct | 8B | 4-bit LoRA |

**LoRA configuration (all LLM notebooks):**
```python
r = 16, lora_alpha = 16, lora_dropout = 0
bias = "none", random_state = 3407
load_in_4bit = True
use_gradient_checkpointing = "unsloth"
finetune_vision_layers = True
finetune_language_layers = True
```

### Zero-Shot Baseline

- **LLaVA-v1.6-Vicuna-13B** — prompted with all 9 Navarasa labels, no fine-tuning

---

## Results

### CLIP & SigLIP (Retrieval Metrics)

| Model | Dataset | Recall@1 | Recall@5 | Recall@10 | Accuracy | Macro-F1 |
|-------|---------|----------|----------|-----------|----------|----------|
| CLIP | Unbalanced | 12.92% | 48.24% | 72.07% | 5.67% | 0.013 |
| CLIP | Balanced | 17.46% | 59.25% | 79.43% | 11.38% | 0.026 |
| SigLIP | Balanced | **28.46%** | **73.22%** | **88.23%** | 11.38% | 0.026 |

> CLIP and SigLIP are optimized for retrieval, not classification. Dataset balancing significantly improved Recall@K by reducing dominant-class bias.

### Baseline Model Misclassifications (Zero-shot)

All zero-shot baseline models failed on Navarasa without cultural fine-tuning:

| Model | Ground Truth | Predicted |
|-------|-------------|-----------|
| Vicuna-7B | Karuna | Hasya |
| LLaVA-1.5-7B | Hasya | Shringara |
| Qwen3-VL-8B | Adbhuta | Shanta |
| Mistral-7B | Bibhatsa | Karuna |
| Florence | Raudra | Shanta |
| Vicuna-13B | Hasya | Unknown |
| Base CLIP | Hasya | Veera |
| Base SigLIP | Bibhatsa | Hasya |

Fine-tuned instruction models (Qwen2.5-VL, Qwen3-VL, Llama 3.2) showed clear improvement after training, correctly identifying emotions that base models missed.

---

## Setup & Requirements

All notebooks run on **Google Colab** (GPU required — T4/A100 recommended).

```bash
# Core dependencies
pip install transformers datasets torch scikit-learn pillow accelerate openpyxl tqdm

# For LoRA fine-tuning (LLMs)
pip install unsloth peft trl bitsandbytes
```

HuggingFace login required:
```python
from huggingface_hub import login
login(token="your_hf_token")
```

---

## Key Findings

- **Cultural context matters** — standard VLMs trained on Western datasets consistently fail on Navarasa labels
- **CLIP/SigLIP are retrieval models**, not classifiers — Recall@K is the right metric, not accuracy/F1
- **Dataset balancing helps retrieval** but does not resolve classification accuracy on its own
- **Instruction-tuned VLMs** (Qwen, Llama) are necessary for true emotion classification
- **SigLIP outperforms CLIP** on retrieval (R@1: 28.46% vs 17.46%) on the balanced dataset

---

## Limitations

- Focused only on visual modality — audio cues are not considered
- Dataset relies on posed expressions rather than spontaneous ones
- Mapping facial expressions to Navarasa involves inherent subjectivity
- Performance is sensitive to dataset quality and class balance

---

## Future Scope

- Incorporate audio-based emotion recognition
- Develop multimodal (audio–visual) emotion models
- Enable real-time emotion recognition from video streams
- Extend the framework to other cultural emotion systems beyond Navarasa

---

## References

- Foteinopoulou & Patras. "EmoClip: A vision-language method for zero-shot video facial expression recognition." IEEE FG 2024.
- Bhattacharyya & Wang. "Evaluating vision-language models for emotion recognition." arXiv:2502.05660, 2025.
- Li et al. "CLIPER: A unified vision-language framework for in-the-wild facial expression recognition." IEEE ICME 2024.
- Happy et al. "The Indian Spontaneous Expression Database for Emotion Recognition." IEEE TAC 8.1, 2015.

---

*Presented by Group 14 · CDAC*



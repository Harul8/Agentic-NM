# RunPod SSH fine-tuning (Qwen3-8B, 48GB VRAM)

Run from your RunPod terminal after SSH into the pod.

## 1) Setup

```bash
cd /workspace
git clone <your-repo-url> nyaymalaw
cd nyaymalaw
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install unsloth transformers trl datasets accelerate peft bitsandbytes sentencepiece
```

## 2) Put latest training data

Make sure these exist and are updated:

- `nyaymalaw_training_examples.md`
- `training/finetune_ready/train.jsonl`
- `training/finetune_ready/val.jsonl`

If needed, regenerate on pod:

```bash
python training/convert_to_finetune.py
```

## 3) Start training in tmux

```bash
tmux new -s nyayma-train
bash training/runpod_finetune_qwen3_8b.sh
```

Detach without stopping:

```bash
Ctrl+b d
```

Reattach:

```bash
tmux attach -t nyayma-train
```

## 4) Monitor

```bash
watch -n 2 nvidia-smi
tail -f training/runpod_train.log
```

## 5) Merge + export GGUF

After training:

```bash
python training/merge_and_export.py
```

Outputs:

- LoRA adapter: `training/lora_model`
- Merged model: `training/merged_model`
- GGUF: `training/nyaymalaw*.gguf`

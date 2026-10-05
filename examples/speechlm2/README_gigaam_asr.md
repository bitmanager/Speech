# GigaAM + Qwen: separate user-transcription head

This offline Russian ASR experiment uses the upstream `DuplexSTTModel`,
`DuplexSTTDataset`, Lightning trainer, autoregressive decoder, and losses.
It is not a reproduction of the paper's causal streaming training recipe.
GigaAM is bidirectional and the initial corpus has no word timestamps.

The integration adds a GigaAM perception adapter and a Balalaika format adapter.
The perception adapter calls official `gigaam.load_model()` and batched
`GigaAM.forward()`. A standard average pool maps the checkpoint's 40 ms frames
to NeMo's 80 ms grid, followed by a linear projection to Qwen's hidden size.
No RNNT head, new training loop, or new ASR loss is introduced.

## Trainable parameters

- Frozen: GigaAM encoder, Qwen backbone, original token embeddings and agent head.
- Trainable: projection and NeMo's independent ASR embeddings/output head.
- With Qwen3-4B-Instruct-2507 and our 768-dimensional GigaAM encoder:
  779,880,960 trainable parameters, including 1,968,640 in the projection.

ASR-only batches have `text_loss_weight: 0` and `duplex_text_channel_weight: 0`.
They do not train the assistant to output silence. They also do not teach it to
answer spoken questions. That requires a subsequent speech-QA/answer dataset.
Both output heads remain present; the untrained agent channel is not an
assistant-quality result at this stage. Russian validation uses raw WER rather
than NeMo's default English text normalizer.

## Dependencies and checkpoints

Upstream baseline: `1688cc3d6a9ade854f544987810c53f605dc86fc`.
Use official GigaAM `7447938d791c4f3e643386ee22c33777004293a5`.
The verified Blackwell environment uses Python 3.12, PyTorch/Torchaudio
2.9.0+cu130, Transformers 4.57.6, FlashAttention 2.8.3, Lightning 2.5.5,
Lhotse 2.0.0a6, and PEFT 0.18.1. FlashAttention's wheel must match PyTorch.
Qwen runs in BF16 with native SDPA; GigaAM uses its official FP16 encoder
autocast and FlashAttention. Spectrogram computation stays FP32 on GPU.

```bash
export QWEN_MODEL_PATH=/llm
export GIGAAM_CHECKPOINT=/model.ckpt
export ASR_DATA_DIR=/data/prepared
export ASR_OUTPUT_DIR=/runs/balalaika

python examples/speechlm2/prepare_balalaika_asr.py \
  --audio-dir /data/selected-source \
  --metadata /data/balalaika_proprietary_v2.parquet \
  --tokenizer "$QWEN_MODEL_PATH" --output "$ASR_DATA_DIR"

python -m pytest --noconftest -s \
  tests/collections/speechlm2/test_gigaam_duplex_integration.py

torchrun --standalone --nproc_per_node=2 examples/speechlm2/stt_duplex_train.py \
  --config-name gigaam_qwen_asr trainer.devices=2 trainer.max_steps=3000

# In another terminal: the unchanged upstream validator on GPU 2.
CUDA_VISIBLE_DEVICES=2 bash examples/speechlm2/validate_gigaam_checkpoints.sh "$ASR_OUTPUT_DIR" 3000
```

The converter joins `shard/key`, removes stress marks, uses actual audio lengths,
and reports all exclusions in `stats.json`. Only mono clips lasting 0.5–20 s
that fit the token/frame capacity are included. No word timestamps are invented.
Identical lowercased transcripts stay in the same train/validation split;
this is not a speaker-disjoint evaluation. File decoding errors are fatal.
Keep the prepared manifest unchanged during a run.

The example saves best and last checkpoints and TensorBoard metrics every
200 optimizer steps. Use a partition with room for full model and optimizer
states. The integration test checks transcription preservation through label
preparation, independent ASR weights, exact freezing, BF16, and nonzero finite
gradients through frozen Qwen to the projection.

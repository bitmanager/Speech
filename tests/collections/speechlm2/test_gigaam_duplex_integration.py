# SPDX-License-Identifier: Apache-2.0
"""Run with the local GigaAM/Qwen checkpoints and a prepared Russian manifest."""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from lhotse import CutSet
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DuplexSTTDataset, DuplexSTTModel


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GIGAAM_CHECKPOINT"), reason="Requires local GigaAM and Qwen checkpoints")
def test_frozen_backbone_transmits_asr_gradient():
    from pathlib import Path

    from nemo.collections.common.data.utils import move_data_to_device

    conf = Path(__file__).resolve().parents[3] / "examples/speechlm2/conf"
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        cfg = compose(config_name="gigaam_qwen_asr")
    model = DuplexSTTModel(OmegaConf.to_container(cfg.model, resolve=True)).to("cuda", dtype=torch.bfloat16)
    model.configure_optimizers()
    dataset = DuplexSTTDataset(
        model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        cfg=cfg.data,
        model_cfg=cfg.model,
    )
    cuts = CutSet.from_file(os.environ["ASR_DATA_DIR"] + "/train.jsonl.gz").subset(first=1)
    batch = move_data_to_device(dataset[cuts], device=model.device)
    prepared = model.prepare_inputs(batch["audio_data"])
    decoded = model.tokenizer.ids_to_text(prepared["asr_labels"][0]).strip()
    assert decoded == batch["audio_data"]["source_texts"][0]
    del prepared
    result = model.training_step(batch, 0)
    assert torch.isfinite(result["loss"]) and result["loss"] > 0
    result["loss"].backward()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        "perception.proj.weight",
        "perception.proj.bias",
        "asr_head.weight",
        "embed_asr_tokens.weight",
    }
    for name, p in model.named_parameters():
        if name in trainable:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
    assert model.asr_head.weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert model.embed_asr_tokens.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
    assert next(model.llm.parameters()).dtype == torch.bfloat16
    model.on_validation_epoch_start()
    assert model.src_wer.normalizer("Двадцать два") == "Двадцать два"
    print({"loss": result["loss"].item(), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from flm_vllm.registration import register_flm_models
from safetensors.torch import save_file

vllm = pytest.importorskip("vllm", reason="CPU vLLM is an optional dependency")
current_platform = importlib.import_module("vllm.platforms").current_platform


@pytest.mark.skipif(
  current_platform.device_type != "cpu",
  reason="requires a vLLM CPU build",
)
def test_cpu_vllm_decode_matches_native_reference_model(
  tmp_path: Path,
  monkeypatch,
) -> None:
  monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", "1")
  monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "auto")
  torch.manual_seed(7)
  config = ReferenceModelConfig(
    vocab_size=32,
    max_seq_len=16,
    # Use a head size supported natively by vLLM's CPU backend. The adapter
    # must not change model dimensions to make an unsupported shape fit.
    d_model=64,
    n_layers=2,
    n_heads=2,
    d_ff=128,
  )
  native = ReferenceModel(config).eval()
  model_dir = tmp_path / "model"
  model_dir.mkdir()
  (model_dir / "config.json").write_text(
    json.dumps(
      {
        "architectures": ["FlmReferenceForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_seq_len,
        "hidden_size": config.d_model,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_heads,
        "head_dim": config.d_model // config.n_heads,
        "intermediate_size": config.d_ff,
        "hidden_act": "silu",
        "rope_theta": config.rope_base,
        "rms_norm_eps": config.norm_eps,
        "attention_bias": config.bias,
        "mlp_bias": config.bias,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
      }
    ),
    encoding="utf-8",
  )
  save_file(
    {
      name: tensor.detach().contiguous().clone()
      for name, tensor in native.state_dict().items()
    },
    model_dir / "model.safetensors",
  )
  prompt = [1, 2, 3]
  expected = _native_greedy_tokens(native, prompt=prompt, count=8)

  register_flm_models()
  llm = vllm.LLM(
    model=str(model_dir),
    tokenizer=None,
    skip_tokenizer_init=True,
    dtype="float32",
    enforce_eager=True,
    max_num_batched_tokens=config.max_seq_len,
  )
  outputs = llm.generate(
    [{"prompt_token_ids": prompt}],
    vllm.SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
  )

  assert list(outputs[0].outputs[0].token_ids) == expected


def _native_greedy_tokens(
  model: ReferenceModel,
  *,
  prompt: list[int],
  count: int,
) -> list[int]:
  tokens = list(prompt)
  generated = []
  with torch.inference_mode():
    for _ in range(count):
      logits, _ = model(torch.tensor([tokens]))
      token = int(logits[0, -1].argmax())
      generated.append(token)
      tokens.append(token)
  return generated

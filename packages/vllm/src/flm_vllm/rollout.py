"""Run rollouts with a vLLM engine."""

from __future__ import annotations

import json
from pathlib import Path

from flm_datasets import get_tokenizer
from flm_train.trainer import RolloutBatch, RolloutSample
from flm_train.types import RolloutPromptConfig


def generate_vllm_rollouts(
  *,
  model_dir: Path,
  encoding_name: str | None = None,
  prompts: tuple[RolloutPromptConfig, ...],
  max_new_tokens: int,
  step: int = 0,
  temperature: float = 0.0,
) -> RolloutBatch:
  try:
    from vllm import LLM, SamplingParams
  except ImportError as exc:  # pragma: no cover - depends on optional vLLM.
    raise RuntimeError(
      "vLLM is not installed. Install vllm in the runtime environment."
    ) from exc

  from flm_vllm.registration import register_flm_models

  register_flm_models()
  encoding_name = resolve_export_encoding_name(
    model_dir=model_dir,
    encoding_name=encoding_name,
  )
  encoding = get_tokenizer(encoding_name)
  llm = LLM(
    model=str(model_dir),
    tokenizer=None,
    skip_tokenizer_init=True,
    trust_remote_code=True,
  )
  sampling = SamplingParams(
    max_tokens=max_new_tokens,
    temperature=temperature,
    logprobs=10,
    prompt_logprobs=1,
  )
  requests = [
    {"prompt_token_ids": encoding.encode_ordinary(prompt.prompt)} for prompt in prompts
  ]
  outputs = llm.generate(requests, sampling)
  samples = [
    _sample_from_output(
      output=output,
      prompt=prompt,
      encoding=encoding,
    )
    for prompt, output in zip(prompts, outputs, strict=True)
  ]
  return RolloutBatch(step=step, samples=tuple(samples))


def resolve_export_encoding_name(*, model_dir: Path, encoding_name: str | None) -> str:
  if encoding_name:
    return encoding_name
  hint_path = Path(model_dir) / "flm_tokenizer.json"
  if not hint_path.is_file():
    raise ValueError(
      "--encoding is required when the model export has no flm_tokenizer.json"
    )
  hint = json.loads(hint_path.read_text(encoding="utf-8"))
  value = hint.get("encoding_name")
  if not isinstance(value, str) or not value:
    raise ValueError(f"invalid tokenizer hint in {hint_path}")
  return value


def _sample_from_output(
  *,
  output,
  prompt: RolloutPromptConfig,
  encoding,
) -> RolloutSample:
  completion = output.outputs[0]
  token_ids = [int(token) for token in completion.token_ids]
  token_texts = [encoding.decode([token]) for token in token_ids]
  top_tokens, top_token_texts, top_log_probs, log_probs = _generated_logprobs(
    completion.logprobs or [],
    token_ids=token_ids,
    encoding=encoding,
  )
  prompt_tokens = tuple(int(token) for token in output.prompt_token_ids)
  prompt_log_probs = tuple(_prompt_logprobs(output.prompt_logprobs or []))
  return RolloutSample(
    name=prompt.name,
    prompt=prompt.prompt,
    prompt_tokens=prompt_tokens,
    prompt_log_probs=prompt_log_probs,
    tokens=tuple(token_ids),
    token_texts=tuple(token_texts),
    log_probs=tuple(log_probs),
    entropy=tuple(),
    top_tokens=tuple(tuple(tokens) for tokens in top_tokens),
    top_token_texts=tuple(tuple(texts) for texts in top_token_texts),
    top_log_probs=tuple(tuple(values) for values in top_log_probs),
    text=prompt.prompt + encoding.decode(token_ids),
  )


def _generated_logprobs(logprobs, *, token_ids: list[int], encoding):
  all_top_tokens = []
  all_top_texts = []
  all_top_log_probs = []
  selected_log_probs = []
  for step_logprobs, token_id in zip(logprobs, token_ids, strict=False):
    ranked = sorted(
      ((int(key), float(value.logprob)) for key, value in step_logprobs.items()),
      key=lambda item: item[1],
      reverse=True,
    )
    all_top_tokens.append([token for token, _ in ranked])
    all_top_texts.append([encoding.decode([token]) for token, _ in ranked])
    all_top_log_probs.append([value for _, value in ranked])
    selected_log_probs.append(float(step_logprobs[token_id].logprob))
  return all_top_tokens, all_top_texts, all_top_log_probs, selected_log_probs


def _prompt_logprobs(prompt_logprobs) -> list[float]:
  values = []
  for item in prompt_logprobs:
    if item is None:
      continue
    if not item:
      continue
    best = max(item.values(), key=lambda value: float(value.logprob))
    values.append(float(best.logprob))
  return values

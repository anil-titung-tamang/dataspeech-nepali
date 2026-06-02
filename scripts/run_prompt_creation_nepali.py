
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs, skip_first_batches
from accelerate.logging import get_logger
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)

logger = get_logger(__name__, log_level="INFO")


# Prompt templates — NEPALI (Devanagari)


# Standard multi-speaker Nepali prompt
NEW_NEPALI_PROMPT = """तपाईंलाई एक व्यक्तिको भाषण नमुनासँग सम्बन्धित छवटा वर्णनात्मक कुञ्जी शब्दहरू दिइनेछन्। यी कुञ्जी शब्दहरूमा समावेश छन्:

१. लिङ्ग (पुरुष, महिला)
२. प्रतिध्वनिको स्तर (धेरै टाढाको आवाज, टाढाको आवाज, अलिकति टाढाको आवाज, अलिकति नजिकको आवाज, धेरै नजिकको आवाज)
३. नमुनामा ध्वनि प्रदूषणको मात्रा (अत्यन्त कोलाहलपूर्ण, धेरै कोलाहलपूर्ण, कोलाहलपूर्ण, अलिकति कोलाहलपूर्ण, लगभग शान्त, धेरै स्पष्ट)
४. वक्ताको आवाजको स्वर (धेरै एकरस, एकरस, अलिकति भावपूर्ण र सजिव, भावपूर्ण र सजिव, धेरै भावपूर्ण र सजिव)
५. वक्ताको बोल्ने गति (धेरै बिस्तारै, बिस्तारै, अलिकति बिस्तारै, मध्यम गति, अलिकति छिटो, छिटो, धेरै छिटो)
६. वक्ताको आवाजको पिच (धेरै कम पिच, कम पिच, अलिकति कम पिच, मध्यम पिच, अलिकति उच्च पिच, उच्च पिच, धेरै उच्च पिच)

यी कुञ्जी शब्दहरू प्रयोग गरेर भाषण नमुनाको सटीक वर्णन गर्ने एक नेपाली पाठ विवरण बनाउनुहोस्।

यदि ध्वनि प्रदूषण 'अत्यन्त कोलाहलपूर्ण' र प्रतिध्वनि 'धेरै टाढाको आवाज' छ भने, 'धेरै खराब रेकर्डिङ' जस्ता शब्द समावेश गर्नुहोस्।
यदि ध्वनि प्रदूषण 'धेरै स्पष्ट' र प्रतिध्वनि 'धेरै नजिकको आवाज' छ भने, 'धेरै राम्रो रेकर्डिङ' जस्ता शब्द समावेश गर्नुहोस्।

'मध्यम गति' र 'मध्यम पिच' जस्ता तटस्थ शब्दहरू आवश्यकताअनुसार हटाउन सकिन्छ।
दिइएका कुञ्जी शब्दहरूभन्दा बाहिर थप विवरण नथप्नुहोस्। शब्दहरूको क्रम परिवर्तन गर्न र समानार्थी शब्द प्रयोग गर्न सकिन्छ।
व्याकरणिक रूपमा सही, सजिलो, र संक्षिप्त एक मात्र विवरण फिर्ता गर्नुहोस्।

उदाहरणका लागि, कुञ्जी शब्दहरू: 'महिला', 'अलिकति टाढाको आवाज', 'कोलाहलपूर्ण', 'धेरै भावपूर्ण र सजिव', 'धेरै बिस्तारै', 'मध्यम पिच' को लागि:
एउटा मान्य विवरण हुन सक्छ: 'एक महिला धेरै बिस्तारै तर धेरै भावपूर्ण ढंगमा बोल्छिन्। रेकर्डिङमा केही कोलाहल र अलिकति प्रतिध्वनि छ।'

कुञ्जी शब्दहरू: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]' को लागि विवरण:
"""

# Single-speaker Nepali prompt (gender removed since speaker is known)
NEW_NEPALI_SINGLE_SPEAKER_PROMPT = """तपाईंलाई [speaker_name] को भाषण नमुनासँग सम्बन्धित चारवटा वर्णनात्मक कुञ्जी शब्दहरू दिइनेछन्:

१. प्रतिध्वनिको स्तर (धेरै टाढाको आवाज, टाढाको आवाज, अलिकति टाढाको आवाज, अलिकति नजिकको आवाज, धेरै नजिकको आवाज)
२. नमुनामा ध्वनि प्रदूषणको मात्रा (अत्यन्त कोलाहलपूर्ण, धेरै कोलाहलपूर्ण, कोलाहलपूर्ण, अलिकति कोलाहलपूर्ण, लगभग शान्त, धेरै स्पष्ट)
३. वक्ताको आवाजको स्वर (धेरै एकरस, एकरस, अलिकति भावपूर्ण र सजिव, भावपूर्ण र सजिव, धेरै भावपूर्ण र सजिव)
४. वक्ताको बोल्ने गति (धेरै बिस्तारै, बिस्तारै, अलिकति बिस्तारै, मध्यम गति, अलिकति छिटो, छिटो, धेरै छिटो)

यी कुञ्जी शब्दहरू प्रयोग गरेर [speaker_name] को भाषण नमुनाको सटीक नेपाली वर्णन बनाउनुहोस्।
दिइएका शब्दहरूभन्दा बाहिर थप विवरण नथप्नुहोस्। एक मात्र संक्षिप्त विवरण फिर्ता गर्नुहोस्।

उदाहरण: कुञ्जी शब्दहरू 'अलिकति टाढाको आवाज', 'लगभग शान्त', 'धेरै भावपूर्ण र सजिव', 'अलिकति छिटो' को लागि:
'[speaker_name] अलिकति छिटो तर धेरै भावपूर्ण ढंगमा बोल्नुहुन्छ। कोठामा हल्का प्रतिध्वनि छ तर पृष्ठभूमि शोर छैन।'

कुञ्जी शब्दहरू: '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]' को लागि विवरण:
"""


# English fallback prompts (from upstream, kept verbatim)


NEW_PROMPT_EN = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:

1. The gender (male, female)
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)

Your task is to create a text description using these keywords that accurately describes the speech sample.
Do not add extra details beyond what has been provided. Only return one concise description.

For the keywords: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', the corresponding description is:
"""

NEW_SINGLE_SPEAKER_PROMPT_EN = """You will be given four descriptive keywords related to an audio sample of [speaker_name]'s speech. These keywords include:

1. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
2. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
3. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
4. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)

Create a concise description for [speaker_name]'s speech. Only return one description.

For the keywords: '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', the corresponding description is:
"""

# Dataclass arguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "HF model name/path for prompt generation (e.g. google/gemma-2-9b-it)."}
    )
    per_device_eval_batch_size: int = field(
        metadata={"help": "Per-device batch size for inference."}
    )
    model_revision: str = field(default="main")
    cache_dir: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default="float16")
    attn_implementation: Optional[str] = field(default="sdpa")
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    use_bnb_nested_quant: Optional[bool] = field(default=False)
    trust_remote_code: Optional[bool] = field(default=False)
    use_fast_tokenizer: Optional[bool] = field(default=True)
    token: Optional[bool] = field(default=True)
    do_sample: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(default=0.6)
    max_new_tokens: Optional[int] = field(default=256)
    torch_compile: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    output_dir: str = field(metadata={"help": "Directory to save the processed dataset."})
    dataset_name: str = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    dataset_split_name: Optional[str] = field(default=None)
    dataset_cache_dir: Optional[str] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    dataloader_num_workers: Optional[int] = field(default=0)
    push_to_hub: Optional[bool] = field(default=False)
    hub_dataset_id: Optional[str] = field(default=None)
    overwrite_output_dir: Optional[bool] = field(default=False)
    save_steps: Optional[int] = field(default=500)
    save_total_limit: Optional[int] = field(default=1)
    speaker_name: Optional[str] = field(default=None)
    is_single_speaker: Optional[bool] = field(default=False)
    is_new_speaker_prompt: Optional[bool] = field(default=False)
    speaker_id_column: Optional[str] = field(default=None)
    speaker_ids_to_name_json: Optional[str] = field(default=None)
    accent_column: Optional[str] = field(default=None)
    # NEW: language of generated descriptions
    prompt_language: Optional[str] = field(
        default="ne",
        metadata={"help": "'ne' for Nepali descriptions (default), 'en' for English."},
    )

    def __post_init__(self):
        if self.push_to_hub and self.hub_dataset_id is None:
            raise ValueError("Specify --hub_dataset_id when using --push_to_hub.")



# Helpers (unchanged from upstream)


def get_quantization_config(model_args):
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    if model_args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_current_device():
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map():
    return {"": get_current_device()} if torch.cuda.is_available() else None


CHECKPOINT_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+).json$")


def save_checkpoint(output_dir, all_generated_ids, step):
    path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}-{step}.json")
    with open(path, "w") as f:
        json.dump([ids.tolist() for ids in all_generated_ids], f)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path) as f:
        data = json.load(f)
    return [np.array(lst) for lst in data]


def sorted_checkpoints(output_dir):
    ordering = []
    for p in Path(output_dir).glob(f"{CHECKPOINT_PREFIX}-*"):
        m = re.match(f".*{CHECKPOINT_PREFIX}-([0-9]+)", str(p))
        if m:
            ordering.append((int(m.groups()[0]), str(p)))
    return [x[1] for x in sorted(ordering)]


def rotate_checkpoints(save_total_limit, output_dir):
    if not save_total_limit or save_total_limit <= 0:
        return
    ckpts = sorted_checkpoints(output_dir)
    for ckpt in ckpts[: max(0, len(ckpts) - save_total_limit)]:
        os.remove(ckpt)


def get_last_checkpoint(folder, return_list=False):
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return [], 0
    ckpts = [p for p in os.listdir(folder) if _RE_CHECKPOINT.search(p)]
    if not ckpts:
        return [], 0
    last = os.path.join(folder, max(ckpts, key=lambda x: int(_RE_CHECKPOINT.search(x).group(1))))
    cur_step = int(re.search(r"checkpoint-(\d+).json", last).group(1))
    if return_list:
        return load_checkpoint(last), cur_step
    return [], cur_step


@dataclass
class DataCollatorWithPadding:
    tokenizer: Any

    def __call__(self, features):
        input_ids = {"input_ids": [f["input_ids"] for f in features]}
        return self.tokenizer.pad(
            input_ids, return_tensors="pt", padding="longest", return_attention_mask=True
        )



# Main


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if data_args.is_single_speaker and data_args.speaker_name is None:
        raise ValueError("`is_single_speaker=True` but `speaker_name` not specified.")

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600 * 3))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    if data_args.overwrite_output_dir and os.path.isdir(data_args.output_dir):
        shutil.rmtree(data_args.output_dir)

    # --- Load dataset ---
    logger.info("*** Loading annotated dataset ***")
    if data_args.dataset_split_name:
        raw_datasets = DatasetDict()
        for split in data_args.dataset_split_name.split("+"):
            with accelerator.local_main_process_first():
                raw_datasets[split] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    num_proc=data_args.preprocessing_num_workers,
                )
    else:
        with accelerator.local_main_process_first():
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )

    raw_datasets_features = set(raw_datasets[next(iter(raw_datasets))].features.keys())

    if data_args.max_eval_samples:
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    EXPECTED_COLUMNS = {"gender", "pitch", "noise", "reverberation", "speech_monotony", "speaking_rate"}
    if data_args.is_single_speaker:
        EXPECTED_COLUMNS = {"noise", "reverberation", "speech_monotony", "speaking_rate"}
    if data_args.is_new_speaker_prompt:
        EXPECTED_COLUMNS.discard("noise")
        EXPECTED_COLUMNS.add("sdr_noise")

    speaker_ids_to_name = {}
    if data_args.speaker_id_column and data_args.speaker_ids_to_name_json:
        EXPECTED_COLUMNS.add(data_args.speaker_id_column)
        with open(data_args.speaker_ids_to_name_json) as f:
            speaker_ids_to_name = json.load(f)

    if not EXPECTED_COLUMNS.issubset(raw_datasets_features):
        missing = EXPECTED_COLUMNS - raw_datasets_features
        raise ValueError(f"Missing columns: {missing}. Dataset has: {raw_datasets_features}")

    # --- Load model ---
    logger.info("*** Loading pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config else None,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        token=model_args.token,
    ).eval()

    if model_args.torch_compile:
        if not callable(getattr(model, "_setup_cache", None)):
            raise ValueError("torch_compile requires a model with static k/v cache (LLaMA, Gemma).")
        model.generation_config.cache_implementation = "static"
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # --- Select prompt templates based on --prompt_language ---
    use_nepali = data_args.prompt_language == "ne"
    logger.info(f"[Nepali] Prompt language: {'Nepali (ne)' if use_nepali else 'English (en)'}")

    base_prompt = NEW_NEPALI_PROMPT if use_nepali else NEW_PROMPT_EN
    single_speaker_prompt = NEW_NEPALI_SINGLE_SPEAKER_PROMPT if use_nepali else NEW_SINGLE_SPEAKER_PROMPT_EN

    speaker_name = data_args.speaker_name
    is_single_speaker = data_args.is_single_speaker
    speaker_id_column = data_args.speaker_id_column
    accent_column_name = data_args.accent_column

    def prepare_dataset(sample):
        sample_prompt = base_prompt
        if is_single_speaker:
            sample_prompt = single_speaker_prompt.replace("[speaker_name]", speaker_name)
        elif speaker_id_column and speaker_ids_to_name.get(str(sample.get(speaker_id_column))):
            name = speaker_ids_to_name[str(sample.get(speaker_id_column))]
            sample_prompt = single_speaker_prompt.replace("[speaker_name]", name)

        for key in EXPECTED_COLUMNS:
            sample_prompt = sample_prompt.replace(f"[{key}]", str(sample.get(key, "")))

        if accent_column_name and sample.get(accent_column_name, "Unindentified") != "Unindentified":
            sample_prompt = sample_prompt.replace("[accent]", sample[accent_column_name])

        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample_prompt}]
        )
        sample["input_ids"] = token_ids
        return sample

    with accelerator.local_main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            num_proc=data_args.preprocessing_num_workers,
            desc="Preparing Nepali prompts",
        )

    model = accelerator.prepare(model)
    data_collator = DataCollatorWithPadding(tokenizer)

    def generate_step(batch):
        output_ids = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=model_args.do_sample,
            temperature=model_args.temperature,
            max_new_tokens=model_args.max_new_tokens,
        )
        return accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)

    def postprocess_dataset(batch):
        prompt_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        generated_texts = tokenizer.batch_decode(batch["generated_ids"], skip_special_tokens=True)
        batch["text_description"] = [
            gen[len(pmt):] for pmt, gen in zip(prompt_texts, generated_texts)
        ]
        return batch

    for split in vectorized_datasets:
        data_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=model_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=data_args.dataloader_num_workers,
            pin_memory=True,
        )
        data_loader = accelerator.prepare(data_loader)
        total_steps = len(data_loader)
        progress_bar = tqdm(range(total_steps), desc=f"Generating [{split}]", disable=not accelerator.is_local_main_process)

        split_output_dir = os.path.join(data_args.output_dir, split)
        all_generated_ids, cur_step = get_last_checkpoint(split_output_dir, accelerator.is_local_main_process)
        accelerator.wait_for_everyone()

        if cur_step > 0:
            logger.info(f"Resuming {split} from step {cur_step}")
            data_loader = skip_first_batches(data_loader, cur_step)
            progress_bar.update(cur_step)

        while cur_step < total_steps:
            for batch in data_loader:
                generated_ids = generate_step(batch)
                generated_ids = accelerator.gather_for_metrics(generated_ids)
                if accelerator.is_local_main_process:
                    all_generated_ids.extend(generated_ids.cpu().numpy())
                cur_step += 1
                progress_bar.update(1)
                if cur_step % data_args.save_steps == 0 or cur_step == total_steps:
                    if accelerator.is_main_process:
                        save_checkpoint(split_output_dir, all_generated_ids, cur_step)
                        rotate_checkpoints(data_args.save_total_limit, output_dir=split_output_dir)
                    accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            vectorized_datasets[split] = vectorized_datasets[split].add_column("generated_ids", all_generated_ids)
        if accelerator.is_main_process:
            vectorized_datasets[split] = vectorized_datasets[split].map(
                postprocess_dataset,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Postprocessing",
                remove_columns=["input_ids", "generated_ids"],
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        vectorized_datasets.save_to_disk(data_args.output_dir)
        if data_args.push_to_hub:
            vectorized_datasets.push_to_hub(
                data_args.hub_dataset_id,
                config_name=data_args.dataset_config_name or "default",
                token=model_args.token,
            )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
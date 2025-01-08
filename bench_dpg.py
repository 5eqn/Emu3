# -*- coding: utf-8 -*-
from PIL import Image
from accelerate import PartialState
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    AutoModelForCausalLM,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from tqdm import tqdm
import torch
import os

from emu3.mllm.processing_emu3 import Emu3Processor


POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
PROMPT_DIR = "../ELLA/dpg_bench/prompts/"
RESULT_DIR = "./results/"

# check directory existence
if not os.path.exists(PROMPT_DIR):
    raise FileNotFoundError(f"Directory {PROMPT_DIR} not found")
os.makedirs(RESULT_DIR, exist_ok=True)

# read all *.txt inside PROMPT_DIR, store into prompt
# prompt is (caption, filename)
prompt = []
for file in os.listdir(PROMPT_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(PROMPT_DIR, file), "r") as f:
            prompt.append((f.read(), file.replace(".txt", "")))
prompt = [(p[0] + POSITIVE_PROMPT, p[1]) for p in prompt]
print(f"Length of prompt: {len(prompt)}")
print(f"First prompt: {prompt[0]}")

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"


# model wrapper
class Emu3Wrapper:
    def load_model(self, device: torch.device):
        """Initialize model here"""

        # prepare model and processor
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            EMU_HUB,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            EMU_HUB, trust_remote_code=True, padding_side="left"
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            VQ_HUB, trust_remote_code=True
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            VQ_HUB, device_map=self.device, trust_remote_code=True
        ).eval()
        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )

    def generate(self, input: list[tuple[str, int]]) -> Image.Image:
        """Generate PIL image for provided caption"""

        classifier_free_guidance = 3.0
        caption = [i[0] for i in input]
        filename = [i[1] for i in input]

        kwargs = dict(
            mode="G",
            ratio="1:1",
            image_area=self.model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        pos_inputs = self.processor(text=caption, **kwargs)
        neg_inputs = self.processor(text=[NEGATIVE_PROMPT] * len(caption), **kwargs)

        # prepare hyper parameters
        GENERATION_CONFIG = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        logits_processor = LogitsProcessorList(
            [
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    classifier_free_guidance,
                    self.model,
                    unconditional_ids=neg_inputs.input_ids.to(self.device),
                ),
                PrefixConstrainedLogitsProcessor(
                    constrained_fn,
                    num_beams=1,
                ),
            ]
        )

        # generate
        outputs = self.model.generate(
            pos_inputs.input_ids.to(self.device),
            GENERATION_CONFIG,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(self.device),
        )

        for idx_i, out in enumerate(outputs):
            mm_list = self.processor.decode(out)
            image_count = 0
            for idx_j, im in enumerate(mm_list):
                if not isinstance(im, Image.Image):
                    continue
                image_count += 1
                if image_count > 1:
                    print(f"More than one image generated for {filename[idx_i]}")
                else:
                    im.save(os.path.join(RESULT_DIR, f"{filename[idx_i]}.png"))


# run inference with distributed state
distributed_state = PartialState()
with distributed_state.split_between_processes(prompt) as prompt:
    BATCH_SIZE = 16
    batches = [prompt[i : i + BATCH_SIZE] for i in range(0, len(prompt), BATCH_SIZE)]
    print(f"Number of batches: {len(batches)}")
    print(f"Batch size: {len(batches[0])}")

    emu3 = Emu3Wrapper()
    emu3.load_model(distributed_state.device)
    for batch in tqdm(batches):
        emu3.generate(batch)

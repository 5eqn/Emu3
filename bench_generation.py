# -*- coding: utf-8 -*-
from PIL import Image
from T2IBenchmark import T2IModelWrapper, calculate_coco_fid
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
import torch

from emu3.mllm.processing_emu3 import Emu3Processor


# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

# prepare input
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."


class Emu3Wrapper(T2IModelWrapper):

    def load_model(self, device: torch.device):
        """Initialize model here"""

        # prepare model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            EMU_HUB,
            device_map="cuda:0",
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
            VQ_HUB, device_map="cuda:0", trust_remote_code=True
        ).eval()
        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )

    def generate(self, caption: str) -> Image.Image:
        """Generate PIL image for provided caption"""

        classifier_free_guidance = 3.0

        kwargs = dict(
            mode="G",
            ratio="1:1",
            image_area=self.model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        pos_inputs = self.processor(text=caption, **kwargs)
        neg_inputs = self.processor(text=NEGATIVE_PROMPT, **kwargs)

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
                    unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
                ),
                PrefixConstrainedLogitsProcessor(
                    constrained_fn,
                    num_beams=1,
                ),
            ]
        )

        # generate
        outputs = self.model.generate(
            pos_inputs.input_ids.to("cuda:0"),
            GENERATION_CONFIG,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to("cuda:0"),
        )

        for idx_i, out in enumerate(outputs):
            mm_list = self.processor.decode(out)
            for idx_j, im in enumerate(mm_list):
                if not isinstance(im, Image.Image):
                    continue
                return im


fid, fid_data = calculate_coco_fid(
    Emu3Wrapper, device="cuda:0", save_generations_dir="coco_generations/"
)

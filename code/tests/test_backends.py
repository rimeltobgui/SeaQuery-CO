"""Exercise native/fusion generation on tiny random CPU models, without downloads."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


class Processor:
    def __init__(self, image_tokens=4, token_types=False):
        self.image_tokens, self.token_types = image_tokens, token_types
        self.tokenizer = self

    def __call__(self, text=None, images=None, **kwargs):
        import torch
        if images is None:
            return NS(input_ids=torch.tensor([[4, 5]], dtype=torch.long))
        ids = torch.tensor([[31] * self.image_tokens + [1, 6, 7]], dtype=torch.long)
        data = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros((1, 3, 8, 8))}
        if self.token_types:
            data["token_type_ids"] = torch.zeros_like(ids)
        return data

    def decode(self, ids, **kwargs):
        return ",".join(str(x) for x in ids.tolist() if x not in (0, 1, 2))


@unittest.skipUnless(AVAILABLE, "Install torch and transformers to exercise tiny-model generation")
class BackendTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "fixture.png"
        Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)).save(self.image)

    def state(self, name, model, processor):
        import torch
        from seaquery.inference.loaders import LoadedModel
        return LoadedModel(name, model.eval(), processor, torch.device("cpu"), torch.float32, {})

    def llava(self):
        from transformers import LlavaConfig, LlavaForConditionalGeneration
        config = LlavaConfig(vision_config={"model_type": "clip_vision_model", "hidden_size": 16,
            "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 8, "patch_size": 4},
            text_config={"model_type": "llama", "vocab_size": 40, "hidden_size": 16,
            "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
            image_token_index=31, image_seq_length=4)
        return self.state("llava", LlavaForConditionalGeneration(config), Processor())

    def paligemma(self):
        from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration
        config = PaliGemmaConfig(vision_config={"model_type": "siglip_vision_model", "hidden_size": 16,
            "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
            "image_size": 8, "patch_size": 4, "projection_dim": 16},
            text_config={"model_type": "gemma", "vocab_size": 40, "hidden_size": 16,
            "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "head_dim": 8,
            "num_key_value_heads": 2, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
            image_token_index=31, projection_dim=16, hidden_size=16, vocab_size=40)
        return self.state("paligemma-mix", PaliGemmaForConditionalGeneration(config), Processor(token_types=True))

    def blip(self, modern):
        from transformers import Blip2Config, Blip2ForConditionalGeneration
        config = Blip2Config(vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 2, "image_size": 8, "patch_size": 4},
            qformer_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 2, "encoder_hidden_size": 16, "cross_attention_frequency": 1},
            text_config={"model_type": "t5", "vocab_size": 40, "d_model": 16, "d_kv": 8,
            "d_ff": 32, "num_layers": 1, "num_decoder_layers": 1, "num_heads": 2,
            "decoder_start_token_id": 0, "eos_token_id": 2, "pad_token_id": 0}, num_query_tokens=2)
        config.image_token_index = 31 if modern else None
        return self.state("blip2", Blip2ForConditionalGeneration(config), Processor(image_tokens=2 if modern else 0))

    def test_llava_native_empty_fusion_equivalence_and_conditioned_fusion(self):
        from seaquery.inference.llava import predict
        state = self.llava()
        self.assertEqual(predict(state, self.image, "Question", max_new_tokens=2),
                         predict(state, self.image, "Question", fusion=True, max_new_tokens=2))
        self.assertIsInstance(predict(state, self.image, "Question", injected="metrics", fusion=True, max_new_tokens=2), str)

    def test_paligemma_noop_and_conditioned_fusion(self):
        from seaquery.inference.paligemma import predict
        state = self.paligemma()
        self.assertEqual(predict(state, self.image, "Question", max_new_tokens=2),
                         predict(state, self.image, "Question", fusion=True, max_new_tokens=2))
        self.assertIsInstance(predict(state, self.image, "Question", injected="metrics", fusion=True, max_new_tokens=2), str)

    def test_blip2_legacy_and_modern_empty_fusion(self):
        from seaquery.inference.blip2 import predict
        for modern in (False, True):
            with self.subTest(modern=modern):
                state = self.blip(modern)
                self.assertEqual(predict(state, self.image, "Question", max_new_tokens=2),
                                 predict(state, self.image, "Question", fusion=True, max_new_tokens=2))
                self.assertIsInstance(predict(state, self.image, "Question", injected="metrics", fusion=True, max_new_tokens=2), str)

    def test_integer_ids_keep_dtype(self):
        import torch
        from seaquery.inference._common import move_inputs
        result = move_inputs({"input_ids": torch.ones((1, 2), dtype=torch.long),
                              "pixel_values": torch.ones((1, 3))}, "cpu", torch.float16)
        self.assertEqual(result["input_ids"].dtype, torch.long)
        self.assertEqual(result["pixel_values"].dtype, torch.float16)

    def test_t4_bf16_rejected_and_cpu_precision(self):
        import torch
        from seaquery.inference.loaders import resolve_runtime
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(ValueError, "Native BF16"):
                resolve_runtime("llava", "cuda", "bfloat16")
            self.assertEqual(resolve_runtime("llava", "cuda", "auto")[1], torch.float16)
            self.assertEqual(resolve_runtime("blip2", "cuda", "auto")[1], torch.float32)
        self.assertEqual(resolve_runtime("llava", "cpu", "auto")[1], torch.float32)

    def test_internvl_tiling_and_cpu_pixels(self):
        import torch
        from seaquery.inference.internvl import dynamic_preprocess, image_pixels
        tiles = dynamic_preprocess(Image.new("RGB", (1792, 448)), max_num=4)
        self.assertEqual(len(tiles), 5)  # four crops and one thumbnail
        self.assertTrue(all(tile.size == (448, 448) for tile in tiles))
        pixels = image_pixels(self.image, "cpu", torch.float32, max_tiles=1)
        self.assertEqual(tuple(pixels.shape), (1, 3, 448, 448))
        self.assertTrue(torch.isfinite(pixels).all())

    def test_local_loader_and_active_lora_without_network(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, CLIPImageProcessor, LlavaProcessor
        from peft import LoraConfig, get_peft_model
        from seaquery.inference.loaders import load_vlm, MODEL_IDS
        from seaquery.inference.llava import predict
        root = Path(self.temp.name)
        model = self.llava().model
        folder = root / "llava"
        model.save_pretrained(folder)
        vocab = {f"t{i}": i for i in range(40)}
        vocab.update({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<image>": 31})
        vocab = {k: v for k, v in vocab.items() if not (k.startswith("t") and v in (0, 1, 2, 3, 31))}
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                            pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
                                            model_input_names=["input_ids", "attention_mask"])
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        processor = LlavaProcessor(image_processor=CLIPImageProcessor(size={"shortest_edge": 8},
                                   crop_size={"height": 8, "width": 8}), tokenizer=tokenizer,
                                   patch_size=4, vision_feature_select_strategy="default")
        processor.save_pretrained(folder)
        lora = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        lora.peft_config["default"].base_model_name_or_path = MODEL_IDS["llava"]
        adapter = root / "adapter"
        lora.save_pretrained(adapter, save_embedding_layers=False)
        with patch("huggingface_hub.file_download._request_wrapper", side_effect=AssertionError("Network used")):
            state = load_vlm("llava", root, adapter=adapter, device="cpu")
            self.assertTrue(hasattr(state.model.language_model.model.layers[0].self_attn.q_proj, "lora_A"))
            self.assertIsInstance(predict(state, self.image, "Question", max_new_tokens=2), str)
            self.assertIsInstance(predict(state, self.image, "Question", injected="metrics", fusion=True, max_new_tokens=2), str)


if __name__ == "__main__":
    unittest.main()

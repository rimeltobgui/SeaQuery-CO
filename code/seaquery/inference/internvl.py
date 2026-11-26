"""InternVL2.5 dynamic tiling and native-template metric fusion."""
import torch
import numpy as np
from PIL import Image
from ._common import open_rgb, insert_metrics, replace_image_embeddings

IV_INSTR = "Answer with a single word or number only. "


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    if not 1 <= min_num <= max_num:
        raise ValueError("Invalid tile limits")
    w, h = image.size
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    best = min(ratios, key=lambda r: (abs(w / h - r[0] / r[1]),
                                     -(w * h > 0.5 * image_size ** 2 * r[0] * r[1])))
    resized = image.resize((image_size * best[0], image_size * best[1]), Image.Resampling.BICUBIC)
    tiles = []
    for n in range(best[0] * best[1]):
        x, y = (n % best[0]) * image_size, (n // best[0]) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


def image_pixels(path, device, dtype, max_tiles=12):
    tiles = dynamic_preprocess(open_rgb(path), max_num=max_tiles)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    tensors = [(torch.from_numpy(np.array(tile, copy=True)).permute(2, 0, 1).float() / 255 - mean) / std
               for tile in tiles]
    return torch.stack(tensors).to(device=device, dtype=dtype)


def subsequence_end(haystack, needle):
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == needle:
            return index + len(needle)
    raise ValueError("Closing image tag missing from InternVL prompt")


@torch.inference_mode()
def predict(state, image, question, injected="", fusion=False, max_new_tokens=16, max_tiles=12):
    model, tokenizer = state.model, state.processor
    pixels = image_pixels(image, state.device, state.dtype, max_tiles)
    if not fusion:
        return model.chat(tokenizer, pixels, "<image>\n" + IV_INSTR + injected + question,
                          dict(max_new_tokens=max_new_tokens, do_sample=False),
                          num_patches_list=[len(pixels)]).strip()
    factory = model.chat.__func__.__globals__.get("get_conv_template")
    if factory is None:
        raise RuntimeError("Pinned InternVL model code must expose get_conv_template")
    template = factory(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], "<image>\n" + IV_INSTR + question)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt().replace(
        "<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token * len(pixels) + "</img>", 1)
    inputs = tokenizer(prompt, return_tensors="pt")
    ids = inputs["input_ids"].to(state.device)
    attention = inputs["attention_mask"].to(state.device)
    embedding = model.language_model.get_input_embeddings()
    embeds = replace_image_embeddings(embedding(ids), ids, model.img_context_token_id,
                                      model.extract_feature(pixels))
    if injected:
        metric_ids = tokenizer(injected, add_special_tokens=False, return_tensors="pt").input_ids.to(state.device)
        position = subsequence_end(ids[0].tolist(), tokenizer("</img>", add_special_tokens=False).input_ids)
        embeds, attention = insert_metrics(embeds, attention, embedding(metric_ids), position)
    output = model.language_model.generate(inputs_embeds=embeds, attention_mask=attention,
                                           max_new_tokens=max_new_tokens, do_sample=False,
                                           eos_token_id=tokenizer.convert_tokens_to_ids(template.sep.strip()))
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

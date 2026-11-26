"""LLaVA native prompting and metric-embedding insertion after image tokens."""
import torch
from ._common import open_rgb, move_inputs, insert_metrics, replace_image_embeddings


@torch.inference_mode()
def predict(state, image, question, injected="", fusion=False, max_new_tokens=32):
    model, processor = state.model, state.processor
    prompt = f"USER: <image>\n{'' if fusion else injected}{question}\nASSISTANT:"
    inputs = move_inputs(processor(text=prompt, images=open_rgb(image), return_tensors="pt"),
                         state.device, state.dtype)
    if not fusion:
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ids = inputs["input_ids"]
    embeds = model.get_input_embeddings()(ids)
    vision = model.vision_tower(inputs["pixel_values"], output_hidden_states=True, return_dict=True)
    features = vision.hidden_states[model.config.vision_feature_layer]
    strategy = model.config.vision_feature_select_strategy
    if strategy == "default":
        features = features[:, 1:]
    elif strategy != "full":
        raise ValueError(f"Unsupported vision selection: {strategy}")
    features = model.multi_modal_projector(features)
    embeds = replace_image_embeddings(embeds, ids, model.config.image_token_index, features)
    attention = inputs["attention_mask"]
    if injected:
        metric_ids = processor.tokenizer(injected, add_special_tokens=False, return_tensors="pt").input_ids.to(state.device)
        position = int(torch.where(ids[0].eq(model.config.image_token_index))[0][-1]) + 1
        embeds, attention = insert_metrics(embeds, attention, model.get_input_embeddings()(metric_ids), position)
    output = model.generate(inputs_embeds=embeds, attention_mask=attention,
                            max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True).strip()

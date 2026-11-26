"""PaliGemma PT/Mix prompting and metric-embedding fusion."""
import torch
from ._common import open_rgb, move_inputs, insert_metrics, replace_image_embeddings


@torch.inference_mode()
def predict(state, image, question, injected="", fusion=False, max_new_tokens=32):
    model, processor = state.model, state.processor
    # Preserve the empty adaptive-fusion no-op from the source implementation.
    if not injected or not str(injected).strip():
        injected, fusion = "", False
    task = "answer en " if state.name == "paligemma-pt" else ""
    text = task + ("" if fusion else injected) + question
    inputs = move_inputs(processor(text=text, images=open_rgb(image), return_tensors="pt", padding=False),
                         state.device, state.dtype)
    if not fusion:
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ids = inputs["input_ids"]
    embeds = model.get_input_embeddings()(ids)
    features = model.get_image_features(inputs["pixel_values"])
    embeds = replace_image_embeddings(embeds, ids, model.config.image_token_index, features)
    metric_ids = processor.tokenizer(str(injected).strip(), add_special_tokens=False,
                                     return_tensors="pt").input_ids.to(state.device)
    if not metric_ids.shape[1]:
        raise ValueError("Metric text produced no tokens")
    metric_embeds = model.get_input_embeddings()(metric_ids)
    position = int(torch.where(ids[0].eq(model.config.image_token_index))[0][-1]) + 1
    embeds, attention = insert_metrics(embeds, inputs["attention_mask"], metric_embeds, position)
    kwargs = dict(inputs_embeds=embeds, attention_mask=attention)
    if "token_type_ids" in inputs:
        types = inputs["token_type_ids"]
        metric_types = torch.zeros(metric_ids.shape, dtype=types.dtype, device=types.device)
        kwargs["token_type_ids"] = torch.cat([types[:, :position], metric_types, types[:, position:]], dim=1)
    output = model.generate(**kwargs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True).strip()

"""BLIP-2 FLAN-T5 inference; support both legacy and image-token processors."""
import torch
from ._common import open_rgb, move_inputs, insert_metrics, replace_image_embeddings


@torch.inference_mode()
def predict(state, image, question, injected="", fusion=False, max_new_tokens=4):
    model, processor = state.model, state.processor
    injected = str(injected or "").strip()
    conditioned = f"{injected} {question}" if injected and not fusion else question
    prompt = f"Question: {conditioned} Answer in one word:"
    inputs = move_inputs(processor(images=open_rgb(image), text=prompt, return_tensors="pt"),
                         state.device, state.dtype)
    if not fusion:
        output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
    else:
        vision = model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True).last_hidden_state
        vision_attention = torch.ones(vision.shape[:-1], dtype=torch.long, device=vision.device)
        query = model.query_tokens.expand(vision.shape[0], -1, -1)
        query_output = model.qformer(query_embeds=query, encoder_hidden_states=vision,
                                     encoder_attention_mask=vision_attention, return_dict=True)
        visual_embeds = model.language_projection(query_output.last_hidden_state)
        embedding = model.get_input_embeddings()
        ids = inputs["input_ids"]
        embeds = embedding(ids)
        attention = inputs["attention_mask"]
        visual_embeds = visual_embeds.to(device=embeds.device, dtype=embeds.dtype)
        token_id = getattr(model.config, "image_token_index", None)
        if token_id is None:
            # Transformers 4.46.3's legacy path prepends Q-Former outputs.
            visual_attention = torch.ones(visual_embeds.shape[:2], dtype=attention.dtype, device=attention.device)
            embeds = torch.cat([visual_embeds, embeds], dim=1)
            attention = torch.cat([visual_attention, attention], dim=1)
            position = visual_embeds.shape[1]
        else:
            positions = torch.where(ids[0].eq(token_id))[0]
            if not torch.equal(positions, torch.arange(visual_embeds.shape[1], device=ids.device)):
                raise ValueError("Expected a contiguous initial BLIP-2 image-token block")
            embeds = replace_image_embeddings(embeds, ids, token_id, visual_embeds)
            position = visual_embeds.shape[1]
        if injected:
            metric_ids = processor.tokenizer(injected, add_special_tokens=False,
                                             return_tensors="pt").input_ids.to(state.device)
            embeds, attention = insert_metrics(embeds, attention, embedding(metric_ids), position)
        output = model.language_model.generate(inputs_embeds=embeds, attention_mask=attention,
                                               do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
    answer = processor.tokenizer.decode(output[0], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True).strip()
    return answer or "[NO_RESPONSE]"

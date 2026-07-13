import torch


def generate_text(
    model,
    tokenizer,
    vision_x,
    prompt,
    device,
    max_new_tokens=16,
    num_beams=4,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
):

    tokenizer.padding_side = "left"

    lang_x = tokenizer(
        [prompt],
        return_tensors="pt",
    )

    vision_x = vision_x.to(device)

    lang_x = {
        k: v.to(device)
        for k, v in lang_x.items()
    }

    with torch.inference_mode():

        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )

    # Prompt 길이
    input_length = lang_x["input_ids"].shape[1]

    # Prompt 제거
    generated_tokens = generated_text[:, input_length:]

    return tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]
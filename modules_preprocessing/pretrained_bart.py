import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from tqdm import tqdm

def bart_summarizer(missing_details_book_id: dict) -> dict:
    device = 0 if torch.cuda.is_available() else -1  # Use CPU if CUDA is not available
    if device == -1:
        print("CUDA is not available, running on CPU.")
    
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    mapping_books = {}
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

    for k, v in tqdm(missing_details_book_id.items(), desc="Summarizing reviews"):
        comprehensive_review = " ".join(v)
        input_length = len(comprehensive_review)
        max_length = min(max(50, int(0.5 * input_length)), 500)  # 50 <= max <= 500

        tokenizer_kwargs = {'truncation': True, 'min_length': 40, 'max_length': max_length}
        try:
            output = pipe(comprehensive_review, **tokenizer_kwargs)
            mapping_books[k] = output[0]['summary_text']
        except Exception as e:
            print(f"Error processing key {k}: {e}")
            mapping_books[k] = None

    return mapping_books
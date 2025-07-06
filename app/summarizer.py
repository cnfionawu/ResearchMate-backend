from transformers import pipeline

summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=-1)  # Force CPU for stability

def summarize(abstracts):
    # Take abstracts and return summaries using T5
    results = []
    for abstract in abstracts:
        try:
            input_text = "summarize: " + abstract
            output = summarizer(input_text, min_length=10, do_sample=False)
            results.append(output[0]['summary_text'])
        except Exception as e:
            print(f"Summarization error: {e}")
            results.append("Summary failed")
    return results


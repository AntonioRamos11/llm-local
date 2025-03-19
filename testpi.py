import requests
import json

# Sample processed paper data
paper_data = {
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely...",
    "content": "Full text of the paper goes here...",
    "sections": {
        "introduction": "Text of introduction...",
        "methodology": "Details of the transformer architecture...",
        "experiments": "Our experiments covered machine translation tasks..."
    },
    "figures": [
        {"id": 1, "caption": "The Transformer Architecture", "description": "Diagram showing encoder and decoder stacks"}
    ],
    "tables": [
        {"id": 1, "caption": "BLEU Scores on WMT 2014", "data": "English-to-German: 28.4, English-to-French: 41.8"}
    ],
    "references": [
        "Vaswani, A., et al. (2017). Attention is all you need. In Advances in neural information processing systems."
    ],
    "output_format": "highlights",  # Options: summary, highlights, complete
    "max_tokens": 1024
}

response = requests.post(
    "http://localhost:5000/api/process_paper",
    json=paper_data,
    headers={"Content-Type": "application/json"}
)

print(json.dumps(response.json(), indent=2))
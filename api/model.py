from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

class JobMatchModel:
    def __init__(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        self.resumes = [
            "Data scientist with expertise in Python, ML, and AI.",
            "Software engineer with experience in cloud computing and backend.",
            "Machine learning engineer specializing in NLP and computer vision."
        ]
        self.res_embeddings = self.embedder.encode(self.resumes)

        dim = self.res_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.res_embeddings)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0
        )

    def match_resumes(self, job_description, top_k=3):
        q_emb = self.embedder.encode([job_description])
        distances, indices = self.index.search(q_emb, top_k)
        results = [(self.resumes[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
        return results

    def chat(self, prompt, max_length=256):
        outputs = self.pipeline(prompt, max_length=max_length, do_sample=True)
        return outputs[0]['generated_text']
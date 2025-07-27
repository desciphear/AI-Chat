import fasttext

class Embedder:
    def __init__(self, model_path="cc.en.300.bin"):
        self.model = fasttext.load_model(model_path)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.model.get_sentence_vector(text).tolist() for text in texts]

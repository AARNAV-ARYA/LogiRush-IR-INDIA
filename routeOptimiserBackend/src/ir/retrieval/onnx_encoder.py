"""Optional local CPU runtime for the same MiniLM encoder."""
from pathlib import Path
import numpy as np


class MiniLMEncoder:
    def __init__(self, directory):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        root = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(root / "model.onnx"), options, providers=["CPUExecutionProvider"])

    def encode(self, texts, **kwargs):
        output = []
        for start in range(0, len(texts), 16):
            rows = self.tokenizer.encode_batch(texts[start:start+16])
            ids = np.asarray([row.ids for row in rows], dtype=np.int64)
            mask = np.asarray([row.attention_mask for row in rows], dtype=np.int64)
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            embeddings = self.session.run(None, {i.name: inputs[i.name] for i in self.session.get_inputs()})[0]
            pooled = (embeddings * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
            output.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))
        return np.vstack(output).astype(np.float32)

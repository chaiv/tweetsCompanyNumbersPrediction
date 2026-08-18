"""Original LSTM teacher augmented with past-quarter recognition and a small adapter."""

import torch

from classifier.LSTMNN import LSTMNN, MEAN_POOLING


class PastOnlyMultitaskTeacher(LSTMNN):
    """Share one representation between financial-class and quarter-ID objectives.

    Top2Vec stays frozen.  A zero-initialized low-rank residual adapter can learn a small
    task-specific change from past data without updating millions of embedding parameters.
    Quarter IDs are defined only over the training quarters of the current rolling fold.
    """

    def __init__(self, emb_size, word_vectors, num_financial_classes, num_training_quarters,
                 pad_token_index, adapter_size=32):
        if num_training_quarters < 2:
            raise ValueError("At least two past quarters are required for quarter recognition")
        super().__init__(
            emb_size=emb_size,
            word_vectors=word_vectors,
            num_classes=num_financial_classes,
            pooling=MEAN_POOLING,
            padTokenIdx=pad_token_index,
            freeze_embeddings=True,
        )
        self.adapter_down = torch.nn.Linear(emb_size, adapter_size, bias=False)
        self.adapter_up = torch.nn.Linear(adapter_size, emb_size, bias=False)
        torch.nn.init.zeros_(self.adapter_up.weight)
        self.quarter_head = torch.nn.Linear(256, num_training_quarters)

    def embed(self, inputs):
        frozen = self.embedding(inputs)
        adapted = self.adapter_up(torch.nn.functional.gelu(self.adapter_down(frozen)))
        return frozen + adapted

    def all_outputs(self, inputs):
        representation = self.encode(inputs)
        return {
            "representation": representation,
            "financial": self.fc3(representation),
            "quarter": self.quarter_head(representation),
        }

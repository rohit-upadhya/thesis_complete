from transformers import Trainer
import torch.nn.functional as F

class CustomTrainer(Trainer):
    def __init__(self, *args, custom_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if self.custom_loss:
            loss = self.custom_loss(outputs, labels)
        else:
            loss = F.cross_entropy(outputs.logits, labels)
        
        return (loss, outputs) if return_outputs else loss
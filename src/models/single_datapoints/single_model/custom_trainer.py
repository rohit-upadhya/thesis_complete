from transformers import Trainer
import torch.nn.functional as F

class CustomTrainer(Trainer):
    def __init__(self, *args, custom_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract the labels from the inputs
        labels = inputs.get("labels")
        # Forward pass to get the model outputs
        outputs = model(**inputs)
        
        # Calculate custom loss if provided, else default to standard loss
        if self.custom_loss:
            loss = self.custom_loss(outputs, labels)
        else:
            # Default to cross entropy loss if no custom loss is provided
            loss = F.cross_entropy(outputs.logits, labels)
        
        return (loss, outputs) if return_outputs else loss
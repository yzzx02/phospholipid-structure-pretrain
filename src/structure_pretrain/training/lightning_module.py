import lightning as L
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.chain_vocab import ChainVocabulary
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.utils.formula import ELEMENTS

from .losses import annotation_metrics, pretraining_loss


class StructurePretrainLightningModule(L.LightningModule):
    def __init__(
        self,
        config: dict,
        chain_tokens: list[str],
        formula_statistics: dict,
        extra_smiles_tokens: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config: ExperimentConfig = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(ExperimentConfig()), config)
        )
        self.chains = ChainVocabulary(chain_tokens)
        self.tokenizer = SmilesTokenizer(extra_smiles_tokens or [])
        self.model = StructurePretrainModel(
            self.config.model, len(self.chains), self.tokenizer, formula_statistics
        )

    def forward(self, batch):
        return self.model(batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # PyG Batch and plain tensors need explicit movement; leave IDs/SMILES on CPU.
        return {
            key: value.to(device) if isinstance(value, (torch.Tensor, Batch)) else value
            for key, value in batch.items()
        }

    def _step(self, batch, stage):
        output = self(batch)
        losses = pretraining_loss(output, batch, self.tokenizer.pad_id, self.config.train)
        values = {**losses, **annotation_metrics(output, batch, self.tokenizer.pad_id)}
        self.log_dict(
            {f"{stage}/{name}": value for name, value in values.items()},
            on_step=stage == "train",
            on_epoch=True,
            batch_size=len(batch["target_architecture"]),
        )
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["representation_metadata"] = {
            "schema": "structure-only-v0.1",
            "elements": list(ELEMENTS),
            "chain_tokens": list(self.chains.tokens),
            "smiles_tokens": list(self.tokenizer.tokens),
            "encoder": self.config.model.encoder,
        }

    def on_load_checkpoint(self, checkpoint):
        expected = {}
        self.on_save_checkpoint(expected)
        if checkpoint.get("representation_metadata") != expected["representation_metadata"]:
            raise ValueError(
                "Checkpoint schema/encoder/vocabulary mismatch; old models incompatible"
            )

# etc

import os
import anndata as ad
import lightning.pytorch as pl
import torch
import anndata as ad
from contextlib import contextmanager
import types

@contextmanager
def allow_untrained(model):
    """
    Temporarily replace `model._check_if_trained` with a no-op so we can call
    methods that normally require the `trained` flag (e.g. get_latent_representation).
    """
    original = model._check_if_trained
    # bound no-op method – keeps `self` argument
    model._check_if_trained = types.MethodType(lambda self, warn=False: None, model)
    try:
        yield
    finally:
        model._check_if_trained = original   # restore immediately

class EpochStepLogger(pl.Callback):
    """
    After each epoch, print:
        • epoch number
        • global training step
        • how many steps the just-finished epoch contained
    """

    def on_train_epoch_end(self, trainer, pl_module):
        steps_per_epoch = trainer.num_training_batches
        epoch          = trainer.current_epoch
        step           = trainer.global_step

        print(f"[epoch {epoch}] finished at global step {step} "
              f"— this epoch had {steps_per_epoch} steps.")

class LatentOnIndexCallback(pl.Callback):
    """
    Capture latent space at chosen training **steps** _or_ **epochs**.

    Parameters
    ----------
    adata : AnnData
        Cells on which to compute the latent representation.
    indices : list[int]
        Numbers of steps *or* epochs to capture (matching `capture_by`).
    capture_by : {"step", "epoch"}
        • "step"  → check `trainer.global_step`  
        • "epoch" → check `trainer.current_epoch`
          (we log only once, at the *last batch* of that epoch).
    subset_idx : slice | list | np.ndarray, optional
        Restrict `adata` to a subset of cells to speed things up.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        indices,
        latent_dir: str,
        latent_base_name: str,
        capture_by: str = "step",
        subset_idx=None,
    ):
        super().__init__()

        if capture_by not in {"step", "epoch"}:
            raise ValueError("capture_by must be 'step' or 'epoch'.")

        self.adata = adata if subset_idx is None else adata[subset_idx]
        self.target_set = set(indices)
        self.latent_dir = latent_dir
        self.latent_base_name = latent_base_name
        self.by = capture_by          # "step" or "epoch"
        self.latents = {}             # {index: tensor}

    def create_anndata_path(self, indice):
        return os.path.join(self.latent_dir, f"{self.latent_base_name}_{self.by}_{indice}.h5ad")
    
    # Lightning requires these even if empty
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def _scvi_model(trainer):
        return trainer._model                         # set by scvi-tools

    @staticmethod
    def _core_module(model):
        return model.module if hasattr(model, "module") else model

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Decide WHETHER to capture at this moment
        if self.by == "step":
            index_now = trainer.global_step

        else:  # capture by "epoch"
            # only fire on the *last* training batch of an epoch
            is_last_batch = (batch_idx + 1) == trainer.num_training_batches
            if not is_last_batch:
                return
            index_now = trainer.current_epoch

        if index_now not in self.target_set:
            return

        # Do the safe forward pass (no grads, no BN updates)
        model = self._scvi_model(trainer)
        core = self._core_module(model)

        was_training = core.training
        core.eval()

        with torch.no_grad(), allow_untrained(model):
            z = model.get_latent_representation(self.adata)
            z_path = self.create_anndata_path(index_now)
            ad.AnnData(X=z).write_h5ad(z_path)

        if was_training:
            core.train()

        self.latents[index_now] = z_path
        print(f"\n[{self.by} {index_now}] latent shape = {z.shape}")

    
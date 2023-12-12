import lightning as L


class LitTransformer(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)
        logits = logits.view(-1, logits.shape[-1])

        y = y.view(-1)
        train_loss = self.loss_fn(logits, y)

        self.log("train_loss", train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logits = logits.view(-1, logits.shape[-1])

        y = y.view(-1)
        val_loss = self.loss_fn(logits, y)

        self.log("val_loss", val_loss, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        return self.optimizer

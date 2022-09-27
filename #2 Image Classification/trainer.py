import torch
from tqdm import tqdm

# compute cross entropy manually since nn.CrossEntropyLoss only accepts labels as target
def cross_entropy(input, target):
    return torch.mean(-torch.sum(target * torch.log(input), 1))


class Trainer:
    def __init__(self, loss_func, optimizer, device='cpu', lr_scheduler=None, summary_writer=None):
        self.device = device

        self.loss_func = loss_func
        self.optimizer = optimizer

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        if summary_writer is not None:
            self.summary_writer = summary_writer

    def fit(self, model, training_loader, validation_loader, num_epochs, evaluate_each_epoch=False):
        model.to(self.device)

        if evaluate_each_epoch:
            self._train(model, training_loader, num_epochs, validation_loader)
        else:
            self._train(model, training_loader, num_epochs)

        if validation_loader is not None:
            self._val(model, validation_loader)

    def _train(
            self,
            model,
            dataloader,
            num_epochs,
            validation_loader=None
    ):
        # Train network
        for epoch in range(num_epochs):

            model.train()
            for batch_idx, (batch_data, batch_labels) in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    ascii=' 123456789#',
                    leave=False,
                    desc=f'#training epoch {epoch + 1}/{num_epochs}',
            ):
                batch_labels = batch_labels.to(self.device)
                batch_data = batch_data.to(self.device)

                # Forward
                output = model(batch_data)
                loss = self.loss_func(output, batch_labels)
                _, prediction = torch.max(output, 1)
                acc = torch.mean((prediction == batch_labels).float())

                if hasattr(self, 'summary_writer'):
                    self.summary_writer.add_scalar("Training loss", loss.item(), epoch)
                    self.summary_writer.add_scalar("Training accuracy", acc, epoch)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Optimize
                self.optimizer.step()

            if hasattr(self, 'exp_lr_scheduler'):
                self.exp_lr_scheduler.step()

            if validation_loader is not None and epoch != num_epochs - 1:
                acc_val, loss_val = self._val(model, validation_loader)
                if hasattr(self, 'summary_writer'):
                    self.summary_writer.add_scalar("Validation loss", loss_val, epoch)
                    self.summary_writer.add_scalar("Validation accuracy", acc_val, epoch)

    def _val(self, model, dataloader):
        model.eval()
        num_correct = 0
        total_loss = 0

        # Evaluate network
        for batch_idx, (batch_data, batch_labels) in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                ascii=' 123456789#',
                leave=False,
                desc=f'evaluating',
        ):
            batch_labels = batch_labels.to(self.device)
            batch_data = batch_data.to(self.device)

            # Forward
            with torch.no_grad():
                output = model(batch_data)
                total_loss += self.loss_func(output, batch_labels).item() * len(output)
                _, prediction = torch.max(output, 1)
                num_correct += (prediction == batch_labels).sum()

        acc = num_correct / len(dataloader.dataset)
        loss = total_loss / len(dataloader.dataset)
        print(f'num_correct: {num_correct}/{len(dataloader.dataset)}, {acc:.5f}')

        return acc, loss

import torch

from configs.config import Config
from torchvision import models
from torch import nn
import lightning as L
import torchmetrics


NUM_CLASSES = 3
def load_object(module_name: str):
    """Загружает объект по его имени (например, строке)."""
    if module_name == 'adam':
        return torch.optim.Adam
    elif module_name == 'sgd':
        return torch.optim.SGD
    elif module_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR
    elif module_name == 'step':
        return torch.optim.lr_scheduler.StepLR
    else:
        raise ValueError(f"Объект '{module_name}' не поддерживается.")

class LModel(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # for optimizer and shaduler
        self._cfg = cfg

        # Модель
        # Загружаем предобученную ResNet18
        self.model = models.resnet18(weights='IMAGENET1K_V1')

        # Замораживаем все слои
        for param in list(self.model.parameters())[:]:
            param.requires_grad = False

        # Заменяем последний слой
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, NUM_CLASSES)
        # Теперь только последние параметры будут обучаться
        for param in self.model.fc.parameters():
            param.requires_grad = True


        # Функция потерь
        self.loss_fn = nn.CrossEntropyLoss() #multilabel
        # self.loss_fn = nn.BCEWithLogitsLoss() #binary

        # Метрики
        self.accuracy = torchmetrics.Accuracy(num_classes=NUM_CLASSES, task='multiclass')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        if labels.dim() > 1:  # проверяем, является ли это one-hot
            labels = torch.argmax(labels, dim=1)  # Преобразуем one-hot в индексы

        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.accuracy(outputs, labels), on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        self.log('val_loss', loss)
        self.log('val_accuracy', self.accuracy(outputs, labels), on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.accuracy(outputs, labels), on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = load_object(self._cfg.optimizer)(
            self.model.parameters(), lr=self._cfg.lr, **self._cfg.optimizer_kwargs,
        )
        scheduler = load_object(self._cfg.scheduler)(
            optimizer, **self._cfg.scheduler_kwargs,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._cfg.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }


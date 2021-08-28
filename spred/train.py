from abc import ABC, abstractmethod
from spred.analytics import Evaluator, ExperimentResult, EpochResult
from spred.util import cudaify
import torch
from tqdm import tqdm


class Trainer(ABC):
    
    def __init__(self, config, criterion, optimizer, train_loader, val_loader,
                 decoder, n_epochs, scheduler, visualizer=None):
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.decoder = decoder
        self.scheduler = scheduler
        self.visualizer = visualizer

    @abstractmethod
    def _epoch_step(self, model):
        ...
    
    def __call__(self, model):
        print("Training with config:")
        print(self.config)
        model = cudaify(model)
        epoch_results = []
        for e in range(1, self.n_epochs+1):
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            if self.scheduler is not None:
                self.scheduler.step()
            eval_result = self.validate_and_analyze(model, e)
            epoch_results.append(EpochResult(e, batch_loss, eval_result))
            print("epoch {}:".format(e))
            print("  training loss: ".format(e) + str(batch_loss))
            print(str(eval_result))
            result = ExperimentResult(self.config, epoch_results)
            # result.show_training_dashboard()
        return model, ExperimentResult(self.config, epoch_results)

    def validate_and_analyze(self, model, epoch):
        model.eval()
        results = list(self.decoder(model, self.val_loader, loss_f=self.criterion))
        validation_loss = self.decoder.get_loss()
        if self.visualizer is not None:
            self.visualizer.visualize(epoch, self.val_loader, results)
        eval_result = Evaluator(results, validation_loss).get_result()
        return eval_result


class SingleTrainer(Trainer):

    def _epoch_step(self, model):
        running_loss = 0.
        denom = 0
        for images, labels in tqdm(self.train_loader, total=len(self.train_loader)):
            self.optimizer.zero_grad()
            output, conf = model(cudaify(images))
            loss = self.criterion(output, conf, cudaify(labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            self.optimizer.step()
            running_loss += loss.item()
            denom += 1
        return running_loss / denom


class PairwiseTrainer(Trainer):

    def _epoch_step(self, model):
        running_loss = 0.
        denom = 0
        for img_x, img_y, lbl_x, lbl_y in tqdm(self.train_loader,
                                               total=len(self.train_loader)):
            self.optimizer.zero_grad()
            output_x, conf_x = model(cudaify(img_x))
            output_y, conf_y = model(cudaify(img_y))
            loss = self.criterion(output_x, output_y, cudaify(lbl_x),
                                  cudaify(lbl_y), conf_x, conf_y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            denom += 1
        return running_loss / denom

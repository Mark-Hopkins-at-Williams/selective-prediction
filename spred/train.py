from abc import ABC, abstractmethod
from spred.analytics import Evaluator, ExperimentResult, EpochResult
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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @abstractmethod
    def _epoch_step(self, model):
        ...
    
    def __call__(self, model):
        print("Training with config:")
        print(self.config)
        model = model.to(self.device)
        epoch_results = []
        for e in range(1, self.n_epochs+1):
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            eval_result = self.validate_and_analyze(model, e)
            epoch_results.append(EpochResult(e, batch_loss, eval_result))
            print("epoch {}:".format(e))
            print("  training loss: ".format(e) + str(batch_loss))
            print(str(eval_result))
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
        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model.train()
            model_out = model(batch, compute_conf=False)
            output, loss, conf = model_out['outputs'], model_out['loss'], model_out['confidences']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            denom += 1
        return running_loss / denom


class PairwiseTrainer(Trainer):
    """ TODO: outdated; FIX! """

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

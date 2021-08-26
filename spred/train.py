from spred.analytics import Evaluator, ExperimentResult, EpochResult
from spred.util import cudaify


class Decoder:

    def __call__(self, net, data):
        raise NotImplementedError("**ABSTRACT METHOD**")


class Trainer:
    
    def __init__(self, config, criterion, optimizer, train_loader, val_loader,
                 decoder, n_epochs, scheduler):
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.decoder = decoder
        self.scheduler = scheduler

    def _epoch_step(self, model):
        raise NotImplementedError("**ABSTRACT METHOD**")
    
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
            eval_result = self.validate_and_analyze(model)
            epoch_results.append(EpochResult(e, batch_loss, eval_result))
            print("epoch {}:".format(e))
            print("  training loss: ".format(e) + str(batch_loss))
            print(str(eval_result))
            result = ExperimentResult(self.config, epoch_results)
            # result.show_training_dashboard()
        return model, ExperimentResult(self.config, epoch_results)

    def validate_and_analyze(self, model):
        model.eval()
        results = list(self.decoder(model, self.val_loader, loss_f=self.criterion))
        validation_loss = self.decoder.get_loss()
        eval_result = Evaluator(results, validation_loss).get_result()
        return eval_result

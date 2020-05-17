# ------------------------------------------------------------------------------
# Agents train the models.
# ------------------------------------------------------------------------------

from ptt.eval.accumulator import Accumulator

class Agent:

    def __init__(self, config, base_criterion, verbose=True):
        """
        :param config: dictionary containing the following training argements:
            - device (gpu id or cpu for training this model)
            - nr_epochs
            - tracking_interval
        """
        self.config = config
        self.device = config.get('device', 'cuda')
        self.base_criterion = base_criterion
        self.metrics = {'loss': self.base_criterion}
        self.verbose = verbose

    def criterion(self, outputs, targets):
        return self.base_criterion(outputs, targets)

    def get_inputs_targets(self, data, model):
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = model.preprocess_input(inputs)       
        return inputs, targets

    def track_statistics(self, epoch, results, model, dataloaders):
        acc = Accumulator(self.metrics.keys())
        for dl_name, dl in dataloaders.items():
            for data in dl:
                inputs, targets = self.get_inputs_targets(data, model)
                outputs = model(inputs)
                for metric_key, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, targets)
                    acc.add(metric_key, metric_value, count=len(inputs))
            for metric_key in self.metrics.keys():
                results.add(epoch=epoch, metric=metric_key, data=dl_name, value=acc.mean(metric_key))
            if self.verbose:
                print('Epoch {} data {} loss {}'.format(epoch, dl_name, acc.mean('accuracy')))

    def train(self, results, model, optimizer, trainloader, valloader=None, dataloaders=dict()):
        """
        :param model: a model instance.
        :param trainloader: dataloader to train the model
        :param dataloaders: dictionary of dataloaders for which results are 
            reported.
        """
        self.track_statistics(0, results, model, dataloaders)
        for epoch in range(self.config.get('nr_epochs', 100)):
            for i, data in enumerate(trainloader):
                # Get data
                inputs, targets = self.get_inputs_targets(data, model)

                # Forward pass
                outputs = model(inputs)

                # Optimization step
                optimizer.zero_grad()
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Track statistics in results
            if (epoch + 1) % self.config.get('tracking_interval', 20) == 0:
                self.track_statistics(epoch + 1, results, model, dataloaders)


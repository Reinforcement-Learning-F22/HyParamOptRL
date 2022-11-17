import torch
from torch import optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from utils import cross_val_fair_scores
from dataset import GermanDataset
from torch.utils.data import DataLoader
import numpy as np
import warnings

warnings.simplefilter("ignore")


def get_optimizer(name):
    if name == 'adam':
        return optim.Adam
    elif name == 'rmsprop':
        return optim.RMSprop
    elif name == 'sgd':
        return optim.SGD
    else:
        raise Exception('Only adam, rmsprop and sgd available')


class Trainer:
    def __init__(self, model, trainer_args):
        """Trainer for adversarial fair representation"""
        self.test_data = None
        self.train_data = None
        self.train_data_ = None
        self.test_data_ = None
        self.device_name = model.device_name
        self.device = torch.device(self.device_name)

        self.epoch = trainer_args.epoch
        self.batch = trainer_args.batch
        self.adv_on_batch = trainer_args.adv_on_batch
        self.model = model
        self.seed = trainer_args.seed
        self.clip_grad = {'ae': trainer_args.grad_clip_ae, 'adv': trainer_args.grad_clip_adv,
                          'class': trainer_args.grad_clip_class}

        # optimizer for autoencoder nets
        self.autoencoder_op = get_optimizer(trainer_args.opt_ae)(
            self.model.autoencoder.parameters(), lr=trainer_args.lr_ae)
        # optimizer for classifier nets
        self.classifier_op = get_optimizer(trainer_args.opt_class)(
            self.model.classifier.parameters(), lr=trainer_args.lr_class)
        # optimizer for adversary nets
        self.adversary_op = get_optimizer(trainer_args.opt_adv)(
            self.model.adversary.parameters(), lr=trainer_args.lr_adv)

        self.name = model.name

    def train_adversary_on_batch(self, batch_data, sensitive_a, label_y):
        """ Train the adversary with fixed classifier-autoencoder """
        # reset gradient
        self.model.classifier.eval()
        self.model.autoencoder.eval()
        self.model.adversary.train()
        self.adversary_op.zero_grad()

        with torch.no_grad():
            reconst, z = self.model.autoencoder(batch_data)
            # predict class label from latent dimension
            pred_y = self.model.classifier(z)

        adv_input = z

        sentive_feature = sensitive_a

        cl_error = self.model.get_class_loss(pred_y, label_y)
        rec_error = self.model.get_recon_loss(reconst, batch_data)

        # predict sensitive attribut from latent dimension
        pred_a = self.model.adversary(adv_input)
        # Compute the adversary loss error
        avd_error = self.model.get_adv_loss(pred_a, sentive_feature)

        # Compute the overall loss and take a negative gradient for the adversary
        error = -self.model.get_loss(rec_error, cl_error, avd_error, label_y)
        error.backward()
        torch.nn.utils.clip_grad_norm(self.model.adversary.parameters(), self.clip_grad['adv'])
        self.adversary_op.step()

        return avd_error

    def train(self, X_train, y_train):
        """Train with fixed adversary or classifier-encoder-decoder across epoch
        """
        self.train_data_ = GermanDataset(X_train[0], y_train, X_train[1])
        self.train_data = DataLoader(self.train_data_, batch_size=self.batch, shuffle=True)
        adversary_loss_log = 0
        total_loss_log = 0
        classifier_loss_log = 0
        autoencoder_loss_log = 0
        torch.autograd.set_detect_anomaly(True)
        for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.train_data):
            train_data = train_x.to(self.device)
            label_y = label_y.to(self.device)
            sensitive_a = sensitive_a.to(self.device)
            self.model.classifier.train()
            self.model.autoencoder.train()
            self.model.adversary.eval()

            # reset the gradients back to zero
            self.autoencoder_op.zero_grad()
            self.classifier_op.zero_grad()

            # compute reconstruction and latent space  the
            reconstructed, z = self.model.autoencoder(train_data)

            # predict class label from Z
            pred_y = self.model.classifier(z)
            adv_input = z
            # compute the adversary loss
            with torch.no_grad():
                # predict sensitive attribute from Z
                pred_a = self.model.adversary(adv_input)  # fixed adversary
                adversary_loss = self.model.get_adv_loss(pred_a, sensitive_a)
            # compute the classification loss
            classifier_loss = self.model.get_class_loss(pred_y, label_y)
            # compute the reconstruction loss
            autoencoder_loss = self.model.get_recon_loss(reconstructed, train_data)
            # compute the total loss
            total_loss = self.model.get_loss(autoencoder_loss, classifier_loss, adversary_loss, label_y)

            # backpropagate the gradient encoder-decoder-classifier with fixed adversary
            total_loss.backward()

            # update parameter of the classifier and the autoencoder
            torch.nn.utils.clip_grad_norm(self.model.autoencoder.parameters(), self.clip_grad['ae'])
            torch.nn.utils.clip_grad_norm(self.model.classifier.parameters(), self.clip_grad['class'])
            self.classifier_op.step()
            self.autoencoder_op.step()

            adversary_loss = 0
            # train the adversary
            for t in range(self.adv_on_batch):
                # print("update adversary iter=", t)
                adversary_loss += self.train_adversary_on_batch(train_data, sensitive_a, label_y)

            adversary_loss = adversary_loss / self.adv_on_batch

            total_loss_log += total_loss.item()
            classifier_loss_log += classifier_loss.item()
            autoencoder_loss_log += autoencoder_loss.item()
            adversary_loss_log += adversary_loss.item()

        # epoch loss
        total_loss_log = total_loss_log / len(self.train_data)
        autoencoder_loss_log = autoencoder_loss_log / len(self.train_data)
        adversary_loss_log = adversary_loss_log / len(self.train_data)
        classifier_loss_log = classifier_loss_log / len(self.train_data)
        return total_loss_log, autoencoder_loss_log, adversary_loss_log, classifier_loss_log

    def test(self):
        adversary_loss_log = 0
        total_loss_log = 0
        classifier_loss_log = 0
        autoencoder_loss_log = 0
        self.model.classifier.eval()
        self.model.autoencoder.eval()
        self.model.adversary.eval()
        with torch.no_grad():
            for n_batch, (test_x, label_y, sensitive_a) in enumerate(self.test_data):
                test_x = test_x.to(self.device)
                label_y = label_y.to(self.device)
                sensitive_a = sensitive_a.to(self.device)
                # compute reconstruction and latent space
                reconstructed, z = self.model.autoencoder(test_x)

                # predict class label from Z
                pred_y = self.model.classifier(z)

                adv_input = z
                # predict sensitive attribute from Z
                pred_a = self.model.adversary(adv_input)  # fixed adversary

                # compute the reconstruction loss
                autoencoder_loss = self.model.get_recon_loss(reconstructed, test_x).item()
                # compute the classification loss
                classifier_loss = self.model.get_class_loss(pred_y, label_y).item()
                # compute the adversary loss
                adversary_loss = self.model.get_adv_loss(pred_a, sensitive_a).item()
                # compute the total loss
                total_loss = self.model.get_loss(autoencoder_loss, classifier_loss, adversary_loss, label_y)

                total_loss_log += total_loss
                classifier_loss_log += classifier_loss
                autoencoder_loss_log += autoencoder_loss
                adversary_loss_log += adversary_loss

            total_loss_log = total_loss_log / len(self.train_data)
            autoencoder_loss_log = autoencoder_loss_log / len(self.train_data)
            adversary_loss_log = adversary_loss_log / len(self.train_data)
            classifier_loss_log = classifier_loss_log / len(self.train_data)
            return total_loss_log, autoencoder_loss_log, adversary_loss_log, classifier_loss_log

    def calc_fair_metrics(self, X_test, y_test, unfair=False):
        self.test_data_ = GermanDataset(X_test[0], y_test, X_test[1])
        self.test_data = DataLoader(self.test_data_, batch_size=self.batch, shuffle=False)
        results = {}
        kfold = KFold(n_splits=5)
        clr = LogisticRegression(max_iter=1000)
        X_test = self.test_data.dataset.X.cpu().detach().numpy()
        y_test = self.test_data.dataset.y.cpu().detach().numpy()
        S_test = self.test_data.dataset.A.cpu().detach().numpy()

        X_transformed = self.model.transform(torch.from_numpy(X_test).to(self.device)).cpu().detach().numpy()
        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed, y_test, kfold, S_test)
        results['test'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                        [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        X_train = self.train_data.dataset.X.cpu().detach().numpy()
        y_train = self.train_data.dataset.y.cpu().detach().numpy()
        S_train = self.train_data.dataset.A.cpu().detach().numpy()
        X_transformed = self.model.transform(torch.from_numpy(X_train).to(self.device)).cpu().detach().numpy()
        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed, y_train, kfold, S_train)
        results['train'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                         [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        if unfair:
            acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_train, y_train, kfold, S_train)
            results['unfair train'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                             [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
            acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_test, y_test, kfold, S_test)
            results['unfair test'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                       [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        return results

    def train_process(self, X_train, y_train):
        for epoch in range(1, self.epoch + 1):  # loop over dataset
            # train
            total_loss_train, autoencoder_loss_train, adversary_loss_train, classifier_loss_train = \
                self.train(X_train, y_train)

        if self.device_name == 'cuda':
            torch.cuda.empty_cache()

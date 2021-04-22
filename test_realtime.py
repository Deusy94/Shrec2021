import numpy as np
import torch
import os

from torch.utils.data import DataLoader

# Import Datasets
from datasets.ShrecRealTime import ShrecRealTime
from models.model_utilizer import ModuleUtilizer

# Import Model
from models.shrec_trasformer import ShrecTrasformer

# Import Utils
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.visualization import plot_confusion_matrix
from utils.state_machine import StateMachine

# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

CLASS = ["ONE", "TWO", "THREE", "FOUR", "OK", "MENU",
         "POINTING", "LEFT", "RIGHT", "CIRCLE", "V",
         "CROSS", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND", "NO_GESTURE"]

class GestureTestRealTime(object):
    """Gesture Recognition Test class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        train_loader (torch.utils.data.DataLoader): Train data loader variable
        val_loader (torch.utils.data.DataLoader): Val data loader variable
        test_loader (torch.utils.data.DataLoader): Test data loader variable
        net (torch.nn.Module): Network used for the current procedure
        lr (int): Learning rate value
        optimizer (torch.nn.optim.optimizer): Optimizer for training procedure
        iters (int): Starting iteration number, not zero if resuming training
        epoch (int): Starting epoch number, not zero if resuming training
        scheduler (torch.optim.lr_scheduler): Scheduler to utilize during training

    """

    def __init__(self, configer):
        self.configer = configer

        self.data_path = configer.get("data", "data_path")      #: str: Path to data directory

        # Train val and test accuracy
        self.accuracy = AverageMeter()

        # DataLoaders
        self.data_loader = None

        # Module load and save utility
        self.device = self.configer.get("device")
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None

        # Training procedure
        self.transforms = None

        # Other useful data
        self.in_planes = np.prod(self.configer.get("data", "n_features"))   #: int: Input channels
        self.window = self.configer.get("data", "n_frames")            #: int: Number of frames per sequence
        self.n_classes = self.configer.get("data", "n_classes")             #: int: Total number of classes for dataset
        self.dataset = self.configer.get("dataset").lower()                 #: str: Type of dataset

    def init_model(self):
        """Initialize model and other data for procedure"""

        # Selecting correct model and normalization variable based on type variable
        if self.dataset == "shrec":
            self.net = ShrecTrasformer(self.in_planes, self.n_classes, n_head=self.configer.get("network", "n_head"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module"))
            Dataset = ShrecRealTime
            self.train_transforms = None
            self.val_transforms = None
        else:
            raise NotImplementedError("Error {}".format(self.dataset))

        self.net, _, _, _ = self.model_utility.load_net(self.net)

        # Setting Dataloaders
        self.data_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="test",
                    transforms=self.transforms, n_frames=self.window),
            batch_size=1, shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __test(self):
        """Testing function."""
        self.net.eval()
        data_list = list()
        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.data_loader, desc="Test")):
                """
                input, gt
                """
                inputs = data_tuple[0].to(self.device)

                tmp = "{};".format(len(data_list) + 1)
                start = list()
                end = list()
                prediction_data = list()
                s = StateMachine(threshold_window_S1=10, threshold_S3=25)
                for idx, window in enumerate(inputs):
                    output = self.net(window.unsqueeze(0))
                    predicted = torch.argmax(output.detach(), dim=-1).cpu().numpy()
                    prediction_data.append(predicted.squeeze())
                    retval = s.update(predicted.squeeze(), idx)
                    if retval:
                        start.append(s.get_start())
                        end.append(s.get_end())
                        pred = np.array(prediction_data)[start[-1]:end[-1] + 1]
                        pred = pred[pred != CLASS.index("NO_GESTURE")]
                        pred = np.bincount(pred.flatten()).argmax()
                        tmp += "{};{};{};".format(CLASS[pred], start[-1] + 2, end[-1] + 2)
                        s.reset()

                if s.active in [2, 3]:
                    if (len(inputs) - 1 - s.get_start()) > 3:
                        start.append(s.get_start())
                        end.append(len(inputs) - 1)
                        pred = np.array(prediction_data)[start[-1]:end[-1] + 1]
                        pred = pred[pred != CLASS.index("NO_GESTURE")]
                        pred = np.bincount(pred.flatten()).argmax()
                        tmp += "{};{};{};".format(CLASS[pred], start[-1] + 2, end[-1] + 2)
                        s.reset()
                tmp += "\n"
                data_list.append(tmp)

        if not os.path.exists("./outputs"):
            os.makedirs("./outputs")
        with open("./outputs/"+self.configer.get("checkpoints", "save_name")+".txt", "wt") as outfile:
            outfile.write("".join(data_list))

    def test(self):
        self.__test()

    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)
        if split == "train" and self.iters % self.save_iters == 0:
            self.tbx_summary.add_scalar('{}_loss'.format(split), self.losses[split].avg, self.iters)
            self.tbx_summary.add_scalar('{}_accuracy'.format(split), self.accuracy[split].avg, self.iters)
            self.losses[split].reset()
            self.accuracy[split].reset()
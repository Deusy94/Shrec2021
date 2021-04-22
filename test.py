import numpy as np
import torch
import os

from torch.utils.data import DataLoader

# Import Datasets
from datasets.Shrec import Shrec
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

class GestureTest(object):
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

        self.frame_error_start = list()
        self.frame_error_end = list()
        self.AP = list()
        self.AR = list()
        self.CORR_CLASS = list()
        self.MISLAB = list()
        self.FALSE_POS = list()
        self.MISSED = list()
        self.confusion = np.zeros((self.configer.get("data", "n_classes") - 1, self.configer.get("data", "n_classes") - 1))

    def init_model(self):
        """Initialize model and other data for procedure"""

        # Selecting correct model and normalization variable based on type variable
        if self.dataset == "shrec":
            self.net = ShrecTrasformer(self.in_planes, self.n_classes,
                                       n_head=self.configer.get("network", "n_head"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module"))
            Dataset = Shrec
            self.train_transforms = None
            self.val_transforms = None
        else:
            raise NotImplementedError("Error {}".format(self.dataset))

        self.net, _, _, _ = self.model_utility.load_net(self.net)

        # Setting Dataloaders
        self.data_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="val",
                    transforms=self.transforms, n_frames=self.window,
                    idx_crossval=self.configer.get("data", "idx_crossval")),
            batch_size=1, shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __test(self):
        """Testing function."""
        self.net.eval()
        data_list = list()
        data_list_gt = list()
        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.data_loader, desc="Test")):
                """
                input, gt
                """
                inputs = data_tuple[0][0].to(self.device)
                gt = data_tuple[1][0].to(self.device)
                start_end = data_tuple[2]

                output = self.net(inputs)

                predicted = torch.argmax(output.detach(), dim=-1)
                correct = gt.detach().squeeze(dim=1)

                self.state_machine(predicted.cpu().numpy(), correct.cpu().numpy(), start_end.cpu().numpy(), data_list, data_list_gt)

        if not os.path.exists("./outputs"):
            os.makedirs("./outputs")
        with open("./outputs/"+self.configer.get("checkpoints", "save_name")+".txt", "wt") as outfile:
            outfile.write("".join(data_list))
        with open("./outputs/"+self.configer.get("checkpoints", "save_name")+"_GT.txt", "wt") as outfile:
            outfile.write("".join(data_list_gt))

        plot_confusion_matrix(self.confusion.astype(np.int16), ["{}".format(i) for i in range(self.n_classes)], False, "./outputs/" + self.configer.get("checkpoints", "save_name") + "_matrix")

        print(f"AP: {np.array(self.AP).mean()}, AR: {np.array(self.AR).mean()}")
        print(f"CORR_CLASS: {np.array(self.CORR_CLASS).mean()}, FALSE_POS: {np.array(self.FALSE_POS).mean()}")

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


    def state_machine(self, predicted, correct, start_end, data_list = None, data_list_gt = None):
        s = StateMachine(threshold_window_S1=10, threshold_S3=25)
        start_gt = list()
        end_gt = list()
        for idx, window in enumerate(correct):
            retval = s.update(window, idx)
            if retval:
                start_gt.append(s.get_start())
                end_gt.append(s.get_end())
                s.reset()
        if s.active in [2, 3]:
            if (len(correct) - 1 - s.get_start()) > 3:
                start_gt.append(s.get_start())
                end_gt.append(len(correct) - 1)

        s.reset()
        start = list()
        end = list()
        for idx, window in enumerate(predicted):
            retval = s.update(window, idx)
            if retval:
                start.append(s.get_start())
                end.append(s.get_end())
                s.reset()
        if s.active in [2, 3]:
            if (len(predicted) - 1 - s.get_start()) > 3:
                start.append(s.get_start())
                end.append(len(predicted) - 1)

        TP = 0
        FP = 0
        couples = dict()
        start_end = start_end[0]
        for i in range(len(start)):
            s = start[i]
            e = end[i]
            best_idx = ((np.abs((np.array(start_end[..., 0]) - s)) + np.abs((np.array(start_end[..., 1]) - e)) / 2)).argmin()
            if best_idx not in couples:
                couples[best_idx] = [i, ((np.abs((np.array(start_end[..., 0]) - s)) + np.abs((np.array(start_end[..., 1]) - e))) / 2).min()]
            else:
                if ((np.abs((np.array(start_end[..., 0]) - s)) + np.abs((np.array(start_end[..., 1]) - e))) / 2).min() < couples[best_idx][1]:
                    couples[best_idx] = [i, ((np.abs((np.array(start_end[..., 0]) - s)) + np.abs((np.array(start_end[..., 1]) - e))) / 2).min()]
                FP += 1

        FN = len(start_gt) - len(couples)
        for k, v in couples.items():
            self.frame_error_start.append(np.abs(start_end[k, 0] - start[v[0]]))
            self.frame_error_end.append(np.abs(start_end[k, 1] - end[v[0]]))
            pred = predicted[start[v[0]]:end[v[0]] + 1]
            pred = pred[pred != CLASS.index("NO_GESTURE")]
            try:
                pred = np.bincount(pred.flatten()).argmax()
            except ValueError:
                print("OK")
            gt = correct[start_end[k][0]:start_end[k][1] + 1]
            gt = gt[gt != CLASS.index("NO_GESTURE")]
            gt = np.bincount(gt.flatten()).argmax()
            self.confusion[gt, pred] += 1
            if pred == gt:
                TP += 1
            else:
                FP += 1

        if data_list is not None:
            tmp = "{};".format(len(data_list) + 1)
            for start, end in zip(start, end):
                pred = predicted[start:end + 1]
                pred = pred[pred != CLASS.index("NO_GESTURE")]
                pred = np.bincount(pred.flatten()).argmax()
                tmp += "{};{};{};".format(CLASS[pred], start+2, end+2)
            tmp += "\n"
            data_list.append(tmp)

        if data_list_gt is not None:
            tmp = "{};".format(len(data_list_gt) + 1)
            for i, (s, e) in enumerate(zip(start_gt, end_gt)):
                gt = correct[s:e + 1]
                gt = gt[gt != CLASS.index("NO_GESTURE")]
                gt = np.bincount(gt.flatten()).argmax()
                tmp += "{};{};{};".format(CLASS[gt], start_end[i, 0], start_end[i, 1])
            tmp += "\n"
            data_list_gt.append(tmp)

        self.AP.append((TP / (TP + FP)) if TP + FP != 0 else 0)
        self.AR.append((TP / (TP + FN)) if TP + FN != 0 else 0)
        self.FALSE_POS.append(FP)
        self.CORR_CLASS.append(TP / len(couples))
        self.MISLAB.append(1 - TP / len(couples))
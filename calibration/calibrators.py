
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

def _histedges_equalN(x, nbin):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def cross_entropy_loss(logits, labels):
    loss = - np.mean(logits[np.arange(len(labels)), labels] )
    return loss


def ece_loss(probs, labels, num_bins = 10, equal_mass = False): 
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = np.equal(predictions, labels)

    if not equal_mass:
        bins = np.linspace(0, 1, num_bins + 1)
    else: 
        bins = _histedges_equalN(confidences, num_bins)

    ece  = 0.0

    for i in range(num_bins):
        in_bin = np.greater_equal(confidences, bins[i]) & np.less(confidences, bins[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(in_bin * accuracies)
            avg_confidence_in_bin = np.mean(in_bin * confidences)
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

# Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
class TempScaling:

    def __init__(self, bias=  False, device=None, print_verbose= False):

        if device is not None: 
            self.device= device
        else: 
            self.device = torch.device('cpu')

        self.temperature = nn.Parameter(torch.ones(1).to(device) * 1.5)
        self.bias = nn.Parameter(torch.ones(1).to(device) * .0) if bias else None

        self.biasFlag = bias
        self.print_verbose = print_verbose

    def forward(self, input):
        return self.temperature_scale(input)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))

        if self.biasFlag: 
            bias = self.bias.unsqueeze(1).expand(logits.size(0), logits.size(1))
            return logits / temperature + bias
        else:     
            return logits / temperature


    def fit(self, probs, labels, eps = 1e-12):

        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs)

        # First: collect all the logits and labels for the validation set   
        before_temperature_nll = cross_entropy_loss(logits, labels)
        before_temperature_ece = ece_loss(probs, labels)
        
        if self.print_verbose:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))


        torch_labels = torch.from_numpy(labels).long().to(self.device)
        torch_logits = torch.from_numpy(logits).float().to(self.device)

        nll_criterion = nn.CrossEntropyLoss()

        # Next: optimize the temperature w.r.t. NLL
        if not self.biasFlag:
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        else:
            optimizer = optim.LBFGS([self.temperature, self.bias], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(torch_logits), torch_labels)
            loss.backward()
            return loss

        loss = -10.0
        new_loss = -1.0 

        while (np.abs(loss - new_loss) > 1e-6):
            loss = new_loss
            optimizer.step(eval)
            
            with torch.no_grad():
                new_loss = nll_criterion(self.temperature_scale(torch_logits), torch_labels)
        
        rescaled_probs = F.softmax(self.temperature_scale(torch_logits), dim=-1).detach().cpu().numpy()

        rescaled_probs = np.clip(rescaled_probs, eps, 1 - eps)


        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = cross_entropy_loss( np.log(rescaled_probs) , labels)
        after_temperature_ece = ece_loss( rescaled_probs,  labels)
        
        if self.print_verbose: 

            print('Optimal temperature: %.3f' % self.temperature.item()) 
            if self.biasFlag:
                print('Optimal bias: %.3f' % self.bias.item())

            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))


    def calibrate(self, probs, eps = 1e-12):
        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs)
        
        torch_logits = torch.from_numpy(logits).float().to(self.device)
        rescaled_probs = F.softmax(self.temperature_scale(torch_logits), dim=-1).detach().cpu().numpy()

        return rescaled_probs
        

class VectorScaling: 

    def __init__(self, num_label, bias=  False, device=None, print_verbose= False):
        
        if device is not None: 
            self.device= device
        else: 
            self.device = torch.device('cpu')

        self.temperature = nn.Parameter(torch.ones(num_label).to(self.device) * 1.5)
        self.bias = nn.Parameter(torch.ones(num_label).to(self.device) * 0.0) if bias else None
        self.biasFlag = bias
        self.print_verbose = print_verbose

    def forward(self, input):
        return self.temperature_scale(input)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), -1)

        if self.biasFlag: 
            bias = self.bias.unsqueeze(0).expand(logits.size(0), -1)
            return logits / temperature + bias
        else:     
            return logits / temperature


    def fit(self, probs, labels, eps = 1e-12):

        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs)

        # First: collect all the logits and labels for the validation set   
        before_temperature_nll = cross_entropy_loss(logits, labels)
        before_temperature_ece = ece_loss(probs, labels)

        if self.print_verbose: 
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))


        torch_labels = torch.from_numpy(labels).long().to(self.device)
        torch_logits = torch.from_numpy(logits).float().to(self.device)

        nll_criterion = nn.CrossEntropyLoss()

        # Next: optimize the temperature w.r.t. NLL
        if not self.biasFlag:
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        else:
            optimizer = optim.LBFGS([self.temperature, self.bias], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(torch_logits), torch_labels)
            loss.backward()
            return loss

        
        loss = -10.0
        new_loss = -1.0 

        while (np.abs(loss - new_loss) > 1e-6):
            loss = new_loss
            optimizer.step(eval)
            
            with torch.no_grad():
                new_loss = nll_criterion(self.temperature_scale(torch_logits), torch_labels)


        rescaled_probs = F.softmax(self.temperature_scale(torch_logits), dim=-1).detach().cpu().numpy()
        rescaled_probs = np.clip(rescaled_probs, eps, 1 - eps)


        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = cross_entropy_loss( np.log(rescaled_probs) , labels)
        after_temperature_ece = ece_loss( rescaled_probs,  labels)
        
        if self.print_verbose:
            print('Optimal temperature: ', self.temperature.detach().cpu().numpy()) 
            if self.biasFlag:
                print('Optimal bias: ' , self.bias.detach().cpu().numpy())

            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))


    def calibrate(self, probs, eps = 1e-12):
        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs)
        
        torch_logits = torch.from_numpy(logits).float().to(self.device)
        rescaled_probs = F.softmax(self.temperature_scale(torch_logits), dim=-1).detach().cpu().numpy()

        return rescaled_probs

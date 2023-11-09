'''
------------------------------------------------------------------------------
                            QUANTUM QUIPU CHALLENGE
------------------------------------------------------------------------------

Team: Quantum Quixotes
Date: 11 September 2023

Description: This code is the solution to the Quantum Quipu Challenge.
             The code is written in Python and uses the Pennylane library
             for the quantum circuit and Pytorch for the optimization.

Solution:    The solution implements a quantum classifier with data reuploading.
'''

import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pennylane as qml
import torch

from torch.autograd import Variable

from prepare_data import Data

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# array of Pauli matrice
Paulis = Variable(torch.zeros([4, 2, 2], dtype=torch.complex128), requires_grad=False)
Paulis[0] = torch.tensor([[0, 1], [1, 0]])          # X
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])       # Y
Paulis[2] = torch.tensor([[1, 0], [0, -1]])         # Z
Paulis[3] = torch.tensor([[1, 0], [0, 1]])          # I


def density_matrix(state):
    '''Returns the density matrix of a state'''
    return torch.ger(state, state.conj())

Observable = Paulis[2] # Z

class DataReuploadingClassifier:
    '''
    Data Reuploading Classifier

    Parameters:
        n_qubits (int): number of qubits
        n_layers (int): number of layers
        optimizer (torch.optim): optimizer
        lr (float): learning rate
        steps (int): number of steps
        batch_size (int): batch size
        Observable (torch.tensor): observable to measure

    '''
    def __init__(self,n_qubits, n_layers, optimizer=torch.optim.Adam, lr=0.1, steps=1024, batch_size=100, Observable=Observable):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * (n_layers + 1)

        self.params = Variable(torch.rand(self.n_params * 3), requires_grad=True)

        self.optimizer = optimizer([self.params], lr=lr)
        self.steps = steps
        self.batch_size = batch_size

        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        self.Observable = Observable

        self.circuit = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x, params, A=None):
        '''
        Quantum circuit for the classifier with data reuploading.
        Each layer is in the form:

            L(theta, x) = U(theta_2) S(x) U_(theta_1)
        
        where U are the trainable gates and S(x) is the data encoding.
        '''
        A = self.Observable if A is None else A
        
        params = params.reshape(self.n_params, 3)

        for n_l in range(self.n_layers + 1):
            for n_q in range(self.n_qubits):
                rx, ry, rz = params[n_l*self.n_qubits + n_q]
                qml.Rot(rx, ry, rz, wires=n_q)

            for n_q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[n_q, n_q + 1])

            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits-1, 0])

            if n_l != self.n_layers:
                [qml.RX(x, wires=n_q) for n_q in range(self.n_qubits)]

        return qml.expval(qml.Hermitian(A, wires=0))
    
    def _cost(self, params, X, y):
        '''
        Cost function for the classifier.
        The cost is calculated using the fidelity between the output state and the target state.

        For label 0 the target state is |0> and for label 1 the target state is |1>.
        '''
        label_0 = torch.tensor([1, 0])
        label_1 = torch.tensor([0, 1])
        state_labels = [label_0, label_1]
        density_matrix_states = [density_matrix(state) for state in state_labels]

        cost = 0.0
        for i in range(len(X)):
            f = self.circuit(X[i], params, density_matrix_states[y[i]])
            cost += torch.abs( (1 - f) ** 2 )
        return cost / len(X)
    
    def _predict(self, X, params=None):
        '''
        Predicts the label of the data.
        '''
        params = self.best_params if params is None else params
        label_0 = torch.tensor([1, 0])
        label_1 = torch.tensor([0, 1])
        state_labels = [label_0, label_1]
        density_matrix_states = [density_matrix(state) for state in state_labels]

        y_pred = []
        for i in range(len(X)):
            fidelities = [self.circuit(X[i], params, state).detach().numpy() for state in density_matrix_states]
            y_pred.append(np.argmax(fidelities))

        return y_pred
    
    def iterate_minibatches(self, inputs, targets, batch_size):
        '''
        Iterates over the data in batches.
        '''
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]
    

    def _fit(self, X, y, X_test, y_test):
        '''
        Fits the classifier to the data.
        '''
        print(qml.draw(self.circuit)(X[0], self.params))

        self.best_params = self.params
        self.best_cost = self._cost(self.params, X, y)
        y_pred = self._predict(X, self.params)
        self.best_acc = accuracy_score(y, y_pred)
        self.best_auc = roc_auc_score(y, y_pred)

        self.best_acc_test = accuracy_score(y_test, self._predict(X_test, self.params))

        self.loss = []
        self.acc = []

        count_without_improvement = 0
        count_lr_changes = 0

        print('Epoch 0 | Cost {:.4f}  Accuracy {:.4f}  AUC {:.4f}  Best Acc {:.4f}  Best AUC {:.4f}'.format(self.best_cost, self.best_acc, self.best_auc, self.best_acc, self.best_auc))

        for n in range(self.steps):
            self.optimizer.zero_grad()

            for X_batch, y_batch in self.iterate_minibatches(X, y, self.batch_size):
                loss = self._cost(self.params, X_batch, y_batch)
                loss.backward()

                self.optimizer.step()

            loss = self._cost(self.params, X, y)
            y_pred = self._predict(X, self.params)
            acc = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_pred)

            y_pred = self._predict(X_test, self.params)
            acc_test = accuracy_score(y_test, y_pred)
            auc_test = roc_auc_score(y_test, y_pred)

            self.loss.append(loss)
            self.acc.append(acc)

            if count_without_improvement > 10:
                self.optimizer.param_groups[0]['lr'] *= 0.5
                print('    Learning rate reduced to {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
                count_without_improvement = 0
                count_lr_changes += 1

            # keeps track of best parameters
            if  acc > self.best_acc:
                self.best_acc = acc
                self.best_params = np.copy(self.params.detach().numpy())
                self.best_auc = auc
                self.best_acc_test = acc_test
                count_without_improvement = 0
            else:
                count_without_improvement += 1

            print(f'Epoch {n+1:3d} | Cost {loss:.4f}  Accuracy {acc:.4f}  AUC {auc:.4f}  Test Acc {acc_test:.4f}  Test AUC {auc_test:.4f}  Best Acc {self.best_acc:.4f}  Best AUC {self.best_auc:.4f} Best Test Acc {self.best_acc_test:.4f}')

            if count_lr_changes > 3:
                print('    Early stopping due to no improvement')
                break

    def save_results(self, path, filename, X, Y):
        '''
        Saves the results of the classifier.
        '''
        if not os.path.exists(path):
            os.mkdir(path)

        with open(path + filename + '.npy', 'wb') as f:
            np.save(f, self.best_params)    # best parameters
        
        self.loss = [loss.detach().numpy() for loss in self.loss]

        dict_resutls = {'loss': self.loss, 'acc': self.acc}
        with open(path + filename + '.pkl', 'wb') as f:
            pickle.dump(dict_resutls, f)  # loss and accuracy

        # Plot results

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot(self.loss, label='Loss', color='red')
        ax.plot(self.acc, label='Accuracy', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss/Accuracy')
        ax.set_title('Loss and Accuracy')
        ax.legend(loc="lower right")

        fig.savefig(path + filename + '.png', dpi = 300)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        x = np.linspace(-np.pi, np.pi, 100)
        y = [self.circuit(x_i, self.best_params) for x_i in x]

        y_pred = self._predict(X)

        ax.plot(x, y, color='red', label=r'\langle Z\rangle', ls='--')
        ax.scatter(X, Y, label='Data', color='blue', marker='x', alpha=0.5)
        ax.scatter(X, y_pred, label='Predicted', color='red', marker='o', alpha=0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Predicted vs Data')

        ax.legend()

        fig.savefig(path + filename + '_pred.png', dpi = 300)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plo Roc curve
        fpr, tpr, _ = roc_curve(Y, y_pred)
        ax.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % self.best_auc)
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        ax.legend(loc="lower right")

        fig.savefig(path + filename + '_roc.png', dpi = 300)


if __name__ == '__main__':

    '''
    ------------------------------------------------------------------------------
                                PREPARE DATA
    ------------------------------------------------------------------------------
    '''
    
    test_data = pd.read_csv('challenge_test.csv', index_col=0)
    train_data = pd.read_csv('challenge_train.csv', index_col=0)

    data_scaled = Data(train_data, test_data)
    data_scaled.prepare_data()

    X_test = data_scaled.X_test_scaled
    y_test = data_scaled.y_test

    X_train = data_scaled.X_train_scaled
    y_train = data_scaled.y


    '''
    ------------------------------------------------------------------------------
                                QUANTUM CLASSIFIER
    ------------------------------------------------------------------------------
    '''

    n_qubits = 2
    n_layers = 2
    steps = 100
    lr = 0.2

    classifier = DataReuploadingClassifier(n_qubits, n_layers, steps=steps, lr=lr)

    X_test = np.array(X_test['F2'])
    y_test = np.array(y_test)

    X = np.array(X_train['F2'])
    y = np.array(y_train)
    classifier._fit(X, y, X_test, y_test)

    y_pred = classifier._predict(X_test)
    
    acc_test = accuracy_score(y_test, y_pred)
    auc_test = roc_auc_score(y_test, y_pred)

    print(f'\nTest Accuracy: {acc_test}')
    print(f'Test AUC: {auc_test}')
    
    path = 'best_params_pytorch/'
    filename = f'nl_{n_layers}_nq_{n_qubits}_lr_{lr}_ep_{steps}_acc_{acc_test:.3f}_auc_{auc_test:.3f}'

    path = path + filename + '/'

    classifier.save_results(path, filename, X_test, y_test)


                    
    


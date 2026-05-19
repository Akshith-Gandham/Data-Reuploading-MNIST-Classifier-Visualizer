import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

nQubits = 4
nSteps = 4
nFeatures = 8


def get_partition(step):
    if step % 2 == 0:
        return [(0, 1), (2, 3)]
    return [(1, 2), (3, 0)]


train_dev = qml.device("lightning.qubit", wires=nQubits, shots=None)


@qml.qnode(train_dev, interface="torch", diff_method="adjoint")
def measurement_circuit(features, weights):
    for i in range(nQubits):
        qml.RY(np.pi * features[i % nFeatures] + weights[0, i], wires=i)
    for step in range(nSteps):
        for i in range(nQubits):
            qml.RY(np.pi * features[i % nFeatures] + weights[step + 1, i], wires=i)
        for c, t in get_partition(step):
            qml.CNOT(wires=[c, t])
    return [qml.expval(qml.PauliZ(i)) for i in range(nQubits)]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, nFeatures)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class QCALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(nSteps + 1, nQubits))

    def forward(self, features):
        results = []
        for i in range(features.size(0)):
            out = measurement_circuit(features[i], self.weights)
            if isinstance(out, (tuple, list)):
                stacked = []
                for o in out:
                    if isinstance(o, torch.Tensor):
                        stacked.append(o)
                    else:
                        stacked.append(torch.tensor(o, dtype=features.dtype))
                out = torch.stack(stacked)
            results.append(out)
        return torch.stack(results, dim=0).to(dtype=features.dtype, device=features.device)


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.qca = QCALayer()
        self.classifier = nn.Linear(nQubits, 10)

    def forward(self, x):
        f = self.encoder(x)
        q = self.qca(f)
        return self.classifier(q)


snap_dev = qml.device("lightning.qubit", wires=nQubits, shots=None)


def make_macro_snapshot_qnode(target_step):
    @qml.qnode(snap_dev, interface="numpy")
    def circ(features, weights):
        for i in range(nQubits):
            qml.RY(np.pi * features[i % nFeatures] + weights[0, i], wires=i)
        for step in range(target_step):
            for i in range(nQubits):
                qml.RY(np.pi * features[i % nFeatures] + weights[step + 1, i], wires=i)
            for c, t in get_partition(step):
                qml.CNOT(wires=[c, t])
        return qml.state()
    return circ


macro_snapshot_qnodes = [make_macro_snapshot_qnode(k) for k in range(nSteps + 1)]


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["modelStateDict"], strict=False)
    return {
        "ldaModels": ckpt["ldaModels"],
        "ldaTestProjections": ckpt["ldaTestProjections"],
        "tsneEmbeddings": ckpt["tsneEmbeddings"],
        "tsneLabels": ckpt["tsneLabels"],
        "macroSnapshotsRaw": ckpt.get("macroSnapshotsRaw")
    }

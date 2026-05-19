import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from model import FullModel, nSteps, macro_snapshot_qnodes

def get_dataloaders(batch_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for batch_idx, (data, target) in enumerate(loader):
        print(batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total += loss.item() * data.size(0)
        n += data.size(0)
    return total / n

def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion(out, target)
            total += loss.item() * data.size(0)
            correct += out.argmax(1).eq(target).sum().item()
            n += data.size(0)
    return total / n, 100.0 * correct / n

def collect_activations(model, loader, n_samples=500, device=torch.device('cpu')):
    model.eval()
    snapshots = [[] for _ in range(nSteps + 1)]
    labels = []
    with torch.no_grad():
        for data, target in loader:
            if len(labels) >= n_samples:
                break
            data = data.to(device)
            for i in range(data.size(0)):
                if len(labels) >= n_samples:
                    break
                feats = model.encoder(data[i:i+1]).squeeze(0).cpu().numpy().astype(np.float64)
                weights = model.qca.weights.detach().cpu().numpy().astype(np.float64)
                for k, q in enumerate(macro_snapshot_qnodes):
                    sv = q(feats, weights)
                    snapshots[k].append(np.concatenate([sv.real, sv.imag]))
                labels.append(target[i].item())
    return {
        'snapshotsPerStep': [np.array(s) for s in snapshots],
        'labels': np.array(labels)
    }

def fit_projections(activations):
    snapshots = activations['snapshotsPerStep']
    labels = activations['labels']
    lda_models, lda_proj, tsne_emb = [], [], []
    for k, X in enumerate(snapshots):
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, labels)
        lda_models.append(lda)
        lda_proj.append(lda.transform(X))
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(X)
        tsne_emb.append(tsne)
    return {
        'ldaModels': lda_models,
        'ldaTestProjections': lda_proj,
        'tsneEmbeddings': tsne_emb,
        'tsneLabels': labels,
        'macroSnapshotsRaw': snapshots
    }

def save_checkpoint(model, projections, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'modelStateDict': model.state_dict(),
        'ldaModels': projections['ldaModels'],
        'ldaTestProjections': projections['ldaTestProjections'],
        'tsneEmbeddings': projections['tsneEmbeddings'],
        'tsneLabels': projections['tsneLabels'],
        'macroSnapshotsRaw': projections.get('macroSnapshotsRaw')
    }, path)

def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['modelStateDict'])
    return {
        'ldaModels': ckpt['ldaModels'],
        'ldaTestProjections': ckpt['ldaTestProjections'],
        'tsneEmbeddings': ckpt['tsneEmbeddings'],
        'tsneLabels': ckpt['tsneLabels']
    }

def main_train(n_epochs=10, batch_size=64, lr=1e-3, save_path='./weights/qca_model.pt'):
    print("Trainng: ")
    device = torch.device('cpu')
    print(device)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = FullModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"trainLoss={train_loss:.4f}  testLoss={test_loss:.4f}  testAcc={test_acc:.2f}%")
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(model.state_dict(), save_path.replace('.pt', f'_epoch{epoch+1}.pt'))

    activations = collect_activations(model, test_loader, n_samples=500)
    projections = fit_projections(activations)
    save_checkpoint(model, projections, save_path)

if __name__ == '__main__':
    main_train(n_epochs=10, batch_size=64, lr=1e-3, save_path='./weights/model.pt')
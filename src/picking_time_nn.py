
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import os

# 1. 데이터 로딩 및 분할 (자동 경로 탐색)
def find_data_file():
    candidates = [
        os.path.join(os.path.dirname(__file__), "picking_time_data.npz"),
        os.path.join(os.path.dirname(__file__), "..", "picking_time_data.npz"),
        os.path.join(os.getcwd(), "picking_time_data.npz"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

# 2. 네트워크 생성 함수
def make_network(input_dim, hidden_layers):
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

# 3. 정규화 함수
def normalize(train, val, test=None):
    mu, std = train.mean(0), train.std(0)
    train_n = (train - mu) / std
    val_n = (val - mu) / std
    if test is not None:
        test_n = (test - mu) / std
        return train_n, val_n, test_n, mu, std
    return train_n, val_n, mu, std

# 4. Cross-validation 함수
def cross_validate(X, y, hidden_layers, epochs=200, batch_size=32, lr=0.01):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_losses, val_losses = [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        # 정규화
        X_tr_n, X_val_n, X_mu, X_std = normalize(X_tr, X_val)[:4]
        y_tr_n, y_val_n, y_mu, y_std = normalize(y_tr.reshape(-1,1), y_val.reshape(-1,1))[:4]
        # DataLoader
        train_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32), torch.tensor(y_tr_n, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val_n, dtype=torch.float32), torch.tensor(y_val_n, dtype=torch.float32))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        # 모델/최적화
        model = make_network(X.shape[1], hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        tr_loss, va_loss = [], []
        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb).squeeze()
                loss = loss_fn(pred, yb.squeeze())
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            tr_loss.append(np.mean(batch_losses))
            # validation
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.tensor(X_val_n, dtype=torch.float32)).squeeze()
                vloss = loss_fn(val_pred, torch.tensor(y_val_n.squeeze(), dtype=torch.float32)).item()
                va_loss.append(vloss)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
    return np.array(train_losses), np.array(val_losses)

# 5. 최종 학습 및 평가 함수
def train_and_evaluate(X_trainval, y_trainval, X_test, y_test, hidden_layers, epochs=200, batch_size=32, lr=0.01):
    # 정규화
    X_tr_n, X_te_n, X_mu, X_std = normalize(X_trainval, X_test)[:4]
    y_tr_n, y_te_n, y_mu, y_std = normalize(y_trainval.reshape(-1,1), y_test.reshape(-1,1))[:4]
    # DataLoader
    train_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32), torch.tensor(y_tr_n, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # 모델/최적화
    model = make_network(X_trainval.shape[1], hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb.squeeze())
            loss.backward()
            optimizer.step()
    # 평가
    model.eval()
    with torch.no_grad():
        y_pred_n = model(torch.tensor(X_te_n, dtype=torch.float32)).squeeze().numpy()
        y_pred = y_pred_n * y_std + y_mu
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    return rmse, y_pred

# 6. Linear baseline
def linear_baseline(X_trainval, y_trainval, X_test, y_test):
    # 정규화
    X_tr_n, X_te_n, X_mu, X_std = normalize(X_trainval, X_test)[:4]
    y_tr_n, y_te_n, y_mu, y_std = normalize(y_trainval.reshape(-1,1), y_test.reshape(-1,1))[:4]
    # lstsq
    w, _, _, _ = np.linalg.lstsq(X_tr_n, y_tr_n, rcond=None)
    y_pred_n = X_te_n @ w
    y_pred = y_pred_n * y_std + y_mu
    rmse = np.sqrt(np.mean((y_pred.squeeze() - y_test) ** 2))
    return rmse, y_pred.squeeze()

# 7. 실행 및 시각화
if __name__ == "__main__":
    # 3-hidden-layer CV
    tr_losses, va_losses = cross_validate(X_trainval, y_trainval, [32, 16, 8])
    mean_tr, mean_va = tr_losses.mean(0), va_losses.mean(0)
    std_va = va_losses.std(0)
    print(f"CV RMSE (last val): {np.sqrt(mean_va[-1]):.3f} ± {np.sqrt(std_va[-1]):.3f}")
    # Plot
    plt.figure()
    for i in range(5):
        plt.plot(va_losses[i], alpha=0.5, label=f"Fold {i+1}")
    plt.plot(mean_tr, label="Mean Train", color="black", linestyle="--")
    plt.plot(mean_va, label="Mean Val", color="red")
    plt.yscale("log")
    plt.legend()
    plt.title("Validation/Train Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.show()

    # 최종 학습/테스트
    nn_rmse, nn_pred = train_and_evaluate(X_trainval, y_trainval, X_test, y_test, [32, 16, 8])
    lin_rmse, lin_pred = linear_baseline(X_trainval, y_trainval, X_test, y_test)
    print(f"Test RMSE (NN): {nn_rmse:.3f} sec")
    print(f"Test RMSE (Linear): {lin_rmse:.3f} sec")

    # Predicted vs Actual plot
    plt.figure()
    plt.scatter(y_test, nn_pred, label="NN", alpha=0.7)
    plt.scatter(y_test, lin_pred, label="Linear", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", label="Perfect")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.title("Predicted vs Actual (Test Set)")
    plt.show()

    # Depth experiment
    depths = [[32], [32, 16], [32, 16, 8]]
    plt.figure()
    for h in depths:
        _, va_losses = cross_validate(X_trainval, y_trainval, h)
        plt.plot(va_losses.mean(0), label=f"{len(h)} layers")
    plt.yscale("log")
    plt.legend()
    plt.title("Depth Experiment: Mean Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.show()

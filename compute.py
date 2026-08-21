import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm  # sometimes needed for joblib lgb.pkl unpickle

from app.model.model import Model
from xgboost import XGBClassifier
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class GatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class MoEPredictor:
    def __init__(self, moe_dir=None):
        if moe_dir is None:
            moe_dir = os.path.join(ART_DIR, "moeModels")
        self.moe_dir = moe_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # preprocessing
        self.scaler = joblib.load(os.path.join(moe_dir, "scaler.pkl"))
        self.label_encoder = joblib.load(os.path.join(moe_dir, "label_encoder.pkl"))

        # classical experts
        self.xgb = XGBClassifier()
        self.xgb.load_model(os.path.join(moe_dir, "xgb.json"))

        self.lgb = joblib.load(os.path.join(moe_dir, "lgb.pkl"))
        self.rf = joblib.load(os.path.join(moe_dir, "rf.pkl"))

        # Replaced CatBoost with GradientBoostingClassifier
        self.gb = joblib.load(os.path.join(moe_dir, "gb.pkl"))

        self.lr = joblib.load(os.path.join(moe_dir, "lr.pkl"))
        self.svm = joblib.load(os.path.join(moe_dir, "svm.pkl"))

        # torch experts
        self.small_mlp = SmallMLP().to(self.device)
        self.small_mlp.load_state_dict(torch.load(os.path.join(moe_dir, "small_mlp.pt"), map_location=self.device))
        self.small_mlp.eval()

        self.deep_mlp = DeepMLP().to(self.device)
        self.deep_mlp.load_state_dict(torch.load(os.path.join(moe_dir, "deep_mlp.pt"), map_location=self.device))
        self.deep_mlp.eval()

        self.gating = GatingNetwork().to(self.device)
        self.gating.load_state_dict(torch.load(os.path.join(moe_dir, "gating.pt"), map_location=self.device))
        self.gating.eval()

        # must match training exactly
        self.FEATURE_COLUMNS = [
            "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
            "TotLen Fwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean",
            "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
            "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std",
            "Flow IAT Max", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Min",
            "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
            "Pkt Len Var", "FIN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
            "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "Down/Up Ratio",
            "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
            "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
            "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
            "Fwd Seg Size Min", "Active Mean", "Active Std",
            "Active Max", "Idle Min"
        ]

    def _dicts_to_matrix(self, rows):
        X = np.zeros((len(rows), len(self.FEATURE_COLUMNS)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(self.FEATURE_COLUMNS):
                X[i, j] = float(r.get(c, 0.0))
        return X

    @torch.no_grad()
    def predict_batch(self, rows):
        """
        rows: list[dict] with raw feature values (must contain FEATURE_COLUMNS keys)
        returns: (labels, probs) where probs is [B,3]
        """
        X = self._dicts_to_matrix(rows)
        Xs = self.scaler.transform(X)

        # classical probs
        outs = [
            self.xgb.predict_proba(Xs),
            self.lgb.predict_proba(Xs),
            self.rf.predict_proba(Xs),
            self.gb.predict_proba(Xs),
            self.lr.predict_proba(Xs),
            self.svm.predict_proba(Xs),
        ]

        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        outs.append(self.small_mlp(Xt).cpu().numpy())
        outs.append(self.deep_mlp(Xt).cpu().numpy())

        expert_outputs = np.stack(outs, axis=1)  # [B,8,3]
        expert_outputs_t = torch.tensor(expert_outputs, dtype=torch.float32, device=self.device)

        gates = self.gating(Xt).unsqueeze(2)  # [B,8,1]
        final_probs = torch.sum(gates * expert_outputs_t, dim=1)  # [B,3]

        pred = torch.argmax(final_probs, dim=1).cpu().numpy()
        probs = final_probs.cpu().numpy()
        labels = self.label_encoder.inverse_transform(pred)
        return labels, probs


class VAEScorer:
    def __init__(self, input_dim=77, hidden_dim=256, latent_dim=16, topk=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.topk = topk

        # model
        self.model = Model(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(self.device)
        ckpt_path = os.path.join(ART_DIR, "model.pth")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

        # threshold
        thr_path = os.path.join(ART_DIR, "threshold.json")
        with open(thr_path, "r") as f:
            self.threshold = float(json.load(f)["threshold"])

        # scaler for RAW -> scaled
        self.scaler = joblib.load(os.path.join(ART_DIR, "scaler_train_benign.pkl"))

    @torch.no_grad()
    def score_scaled(self, X_scaled_np: np.ndarray):
        xb = torch.from_numpy(X_scaled_np.astype(np.float32)).to(self.device, non_blocking=True)
        recon, mu, logvar = self.model(xb)

        se = (recon - xb) ** 2
        scores = se.sum(dim=1)
        flags = scores > self.threshold

        topv, topi = torch.topk(se, k=min(self.topk, se.size(1)), dim=1)

        return {
            "threshold": float(self.threshold),
            "scores": scores.detach().cpu().numpy(),
            "flags": flags.detach().cpu().numpy().astype(bool),
            "top_idx": topi.detach().cpu().numpy(),
            "top_sqerr": topv.detach().cpu().numpy(),
        }

    def score_raw(self, X_raw_np: np.ndarray):
        X_scaled_np = self.scaler.transform(X_raw_np).astype(np.float32)
        return self.score_scaled(X_scaled_np)


class ThreatPipeline:
    def __init__(self, vae_scorer, moe_predictor, vae_feature_names):
        self.vae = vae_scorer
        self.moe = moe_predictor
        self.vae_feature_names = vae_feature_names  # list length 77

    def _dicts_to_matrix(self, rows, cols):
        X = np.zeros((len(rows), len(cols)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                X[i, j] = float(r.get(c, 0.0))
        return X

    def predict_batch(self, rows):
        X_vae_raw = self._dicts_to_matrix(rows, self.vae_feature_names)
        vae_out = self.vae.score_raw(X_vae_raw)

        flags = vae_out["flags"]
        scores = vae_out["scores"]

        attack_label = np.array(["BENIGN"] * len(rows), dtype=object)
        attack_conf = np.zeros(len(rows), dtype=np.float32)

        if flags.any():
            flagged_rows = [rows[i] for i, f in enumerate(flags) if f]
            labels, probs = self.moe.predict_batch(flagged_rows)  # probs: [K,3]
            confs = probs.max(axis=1)

            k = 0
            for i, f in enumerate(flags):
                if f:
                    attack_label[i] = labels[k]
                    attack_conf[i] = float(confs[k])
                    k += 1

        results = []
        for i in range(len(rows)):
            idxs = vae_out["top_idx"][i].tolist()
            sqe = vae_out["top_sqerr"][i].tolist()
            contrib = [
                {"feature": self.vae_feature_names[j], "sq_error": float(v)}
                for j, v in zip(idxs, sqe)
            ]

            results.append({
                "is_anomaly": bool(flags[i]),
                "score": float(scores[i]),
                "threshold": float(vae_out["threshold"]),
                "attack_type": str(attack_label[i]),
                "attack_confidence": float(attack_conf[i]),
                "top_contributors": contrib,
            })

        return results


vae_names_path = os.path.join(ART_DIR, "features_name.json")
with open(vae_names_path, "r") as f:
    meta = json.load(f)

vae_feature_names = meta["feature_names"]
assert len(vae_feature_names) == 77, f"Expected 77, got {len(vae_feature_names)}"

vae = VAEScorer()
moe = MoEPredictor(moe_dir=os.path.join(ART_DIR, "moeModels"))
PIPELINE = ThreatPipeline(vae, moe, vae_feature_names)


'''
All below functions are for prod to be called by api.
'''


def _missing_keys(row, required):
    return [k for k in required if k not in row]


def validate_rows(rows):
    vae_missing = {}
    moe_missing = {}

    for i, r in enumerate(rows):
        mv = _missing_keys(r, vae_feature_names)
        mm = _missing_keys(r, PIPELINE.moe.FEATURE_COLUMNS)
        if mv: vae_missing[i] = mv[:10]
        if mm: moe_missing[i] = mm[:10]

    return {"vae_missing": vae_missing, "moe_missing": moe_missing}


def predict(rows): # For API, the API can call this function
    """
        rows: list[dict] raw CICIDS features
        returns: json of results and input_issues
    """
    issues = validate_rows(rows)
    results = PIPELINE.predict_batch(rows)
    return {"results": results, "input_issues": issues}

"""
Bayesian Hyperparameter Optimization - Optimiert auf Exact Match
Mit SQLite-Storage (crash-sicher) und Memory Cleanup
"""

import os
import gc
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from nptdms import TdmsFile

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import optuna

# --- TF Warnungen reduzieren ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- GPU Memory Growth (verhindert OOM) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# --- Logging konfigurieren ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- Config ---
tdms_folder = r"data"
kraft_channel = "K3_Ch1_Mod2/AI6"
trigger_channel = "Trigger_Mod1/DI0"
min_length = 100
output_dir = r"hp_search/output"
os.makedirs(output_dir, exist_ok=True)
def get_v_number(name: str):
    name = name.lower()
    parts = name.split("_")

    v_part = next(
        part for part in parts
        if part.startswith("v") and part[1:].isdigit()
    )

    return int(v_part[1:])

def label_from_name_multi(name: str):
    name = name.lower()
    parts = name.split("_")

    try:
        t_part = next(p for p in parts if p.startswith("t") and p[1:].isdigit())
        a_part = next(p for p in parts if p.startswith("a") and p[1:].isdigit())

        t_val = int(t_part[1:])  # 1–3
        a_val = int(a_part[1:])  # 1–3

        T = np.zeros(3, dtype=int)
        A = np.zeros(3, dtype=int)

        T[t_val - 1] = 1
        A[a_val - 1] = 1

        return T, A

    except:
        raise ValueError(f"Fehler im Label: {name}")

# --- TDMS-Dateien einlesen ---
# V0 wird als Referenz mitgenommen, V1–V9 als Trainingsdaten
tdms_files = []

for fn in os.listdir(tdms_folder):
    if not fn.lower().endswith(".tdms"):
        continue

    try:
        v_num = get_v_number(fn)

        if v_num == 0:
            tdms_files.append((os.path.join(tdms_folder, fn), v_num))

        elif 1 <= v_num <= 9:
            tdms_files.append((os.path.join(tdms_folder, fn), v_num))

        else:
            logging.warning(f"Übersprungen: {fn} (V-Nummer nicht erlaubt)")

    except Exception as e:
        logging.warning(f"Übersprungen: {fn} ({e})")

# --- Hub-Segmentierung & Vorbereitung ---
X_hubs = []        # V1 bis V9
X_hubs_ref = []    # V0 Referenz / Leerhub

y_T = []
y_A = []

hub_lengths = []
hub_lengths_ref = []

for path, v_num in tdms_files:
    tdms = TdmsFile.read(path)
    grp = tdms.groups()[0].name

    sig_force = tdms[grp][kraft_channel].data
    sig_trig  = (tdms[grp][trigger_channel].data > 0.5).astype(int)

    diff = np.diff(np.concatenate(([0], sig_trig)))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(ends) < len(starts):
        ends = np.append(ends, len(sig_trig))

    for s, e in zip(starts, ends):
        if e - s < min_length:
            continue

        hub = sig_force[s:e]

        if v_num == 0:
            X_hubs_ref.append(hub)
            hub_lengths_ref.append(len(hub))
        else:
            X_hubs.append(hub)

            T, A = label_from_name_multi(os.path.basename(path))
            y_T.append(T)
            y_A.append(A)

            hub_lengths.append(len(hub))

logging.info(f"Anzahl V1–V9-Hübe: {len(X_hubs)}")
logging.info(f"Anzahl V0-Referenzhübe: {len(X_hubs_ref)}")

if len(X_hubs) == 0:
    raise ValueError("Es wurden keine gültigen V1–V9-Hübe extrahiert.")

if len(X_hubs_ref) == 0:
    raise ValueError("Es wurden keine gültigen V0-Referenzhübe extrahiert.")

logging.info(f"Maximale V1–V9-Hub-Länge: {max(hub_lengths)}")
logging.info(f"Minimale V1–V9-Hub-Länge: {min(hub_lengths)}")
logging.info(f"Maximale V0-Hub-Länge: {max(hub_lengths_ref)}")
logging.info(f"Minimale V0-Hub-Länge: {min(hub_lengths_ref)}")


# --- X-/Y-Alignment für V0 und V1–V9 gemeinsam ---
# Wichtig: V0 und V1–V9 bekommen exakt dieselbe gemeinsame Hublänge.

all_hubs_for_length = X_hubs + X_hubs_ref
half_lengths = []

for hub in all_hubs_for_length:
    center = len(hub) // 2
    left = center
    right = len(hub) - center
    half_len = min(left, right)
    half_lengths.append(half_len)

common_half_len = min(half_lengths)

if common_half_len < 1:
    raise ValueError("Mindestens ein Hub ist zu kurz für mittiges X-Alignment.")

def center_cut_and_y_align(hubs, common_half_len):
    aligned_hubs = []

    for hub in hubs:
        center = len(hub) // 2
        start_cut = center - common_half_len
        end_cut = center + common_half_len

        hub_x_aligned = hub[start_cut:end_cut]

        # Y-Alignment: letzter Punkt jedes Hubs wird auf 0 gesetzt
        hub_xy_aligned = hub_x_aligned - hub_x_aligned[-1]

        aligned_hubs.append(hub_xy_aligned)

    return aligned_hubs


X_aligned = center_cut_and_y_align(X_hubs, common_half_len)
X_ref_aligned = center_cut_and_y_align(X_hubs_ref, common_half_len)

logging.info(f"Gemeinsame Hublänge nach X-/Y-Alignment: {len(X_aligned[0])}")


# --- V0-Mittelwertshub bilden ---
X_ref_matrix = np.vstack(X_ref_aligned)
mean_ref_hub = np.mean(X_ref_matrix, axis=0)

logging.info(f"V0-Mittelwertshub gebildet aus {X_ref_matrix.shape[0]} Hüben.")


# --- V0-Mittelwertshub von jedem V1–V9-Hub abziehen ---
X_corrected = []

for hub in X_aligned:
    hub_corrected = hub - mean_ref_hub
    X_corrected.append(hub_corrected)


# Prüfen der finalen Hublänge
hub_lengths_unique = sorted(set(len(hub) for hub in X_corrected))
logging.info(f"Finale Hublängen nach V0-Abzug: {hub_lengths_unique}")

if len(hub_lengths_unique) != 1:
    raise ValueError("Nach Alignment und V0-Abzug sind die Hubs nicht gleich lang.")
# --- Format für CNN ---
X = np.array(X_corrected, dtype='float32')[..., np.newaxis]

y_T = np.array(y_T, dtype='float32')
y_A = np.array(y_A, dtype='float32')

# --- Normierung pro Hub ---
den = np.max(np.abs(X), axis=1, keepdims=True)
den[den == 0] = 1.0
X = X / den

# --- Stratify-Label aus T- und A-Klasse bilden ---
# T1/A1 -> 0, T1/A2 -> 1, ..., T3/A3 -> 8
y_cls = np.argmax(y_T, axis=1) * 3 + np.argmax(y_A, axis=1)

# --- Train/Test-Split mit korrektem stratify ---
X_trainval, X_test, yT_trainval, yT_test, yA_trainval, yA_test, y_trainval_cls, _ = train_test_split(
    X, y_T, y_A, y_cls,
    test_size=0.2,
    stratify=y_cls,
    random_state=42
)

y_trainval_int = y_trainval_cls

logging.info(f"TrainVal: {X_trainval.shape}, Test: {X_test.shape}")
logging.info(f"yT TrainVal: {yT_trainval.shape}, yA TrainVal: {yA_trainval.shape}")
# =============================================================================
# CNN MIT ERWEITERTEM SUCHRAUM + SIGNAL-GRÖSSENPRÜFUNG
# =============================================================================
def build_model_optuna(trial, input_shape):

    n_conv = trial.suggest_int("n_conv_blocks", 2, 7)

    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
    pool_size     = trial.suggest_categorical("pool_size", [2, 4, 5])
    dropout       = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    lr            = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    l2_lambda     = trial.suggest_float("l2_lambda", 1e-7, 1e-3, log=True)

    n_dense = trial.suggest_int("n_dense_layers", 1, 5)
    dense_units = trial.suggest_categorical("dense_units", [64, 128, 256, 512, 1024, 2048])

    signal_len = input_shape[0]
    max_kernel = 7

    for i in range(n_conv):
        signal_len = (signal_len - max_kernel + 1) // pool_size

    if signal_len < 10:
        raise optuna.TrialPruned(
            f"Signal zu klein: {signal_len} < 10 nach {n_conv} Conv-Blöcken mit Pool {pool_size}"
        )

    kernel_reg = regularizers.l2(l2_lambda)

    filter_spaces = {
        1: [16, 32, 48],
        2: [32, 48, 64],
        3: [48, 64, 96],
        4: [64, 96, 128],
        5: [96, 128, 160],
        6: [128, 160, 192],
        7: [160, 192, 256],
    }

    inp = Input(shape=input_shape)
    x = inp

    for i in range(n_conv):
        block_idx = i + 1
        filters = trial.suggest_categorical(f"filters_block_{block_idx}", filter_spaces[block_idx])
        kernel_size = trial.suggest_categorical(f"kernel_block_{block_idx}", [3, 5, 7])
        conv_name = "target_conv" if block_idx == n_conv else None

        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            kernel_regularizer=kernel_reg,
            name=conv_name
        )(x)

        if use_batchnorm:
            x = BatchNormalization()(x)

        x = MaxPooling1D(pool_size)(x)
        x = Dropout(dropout)(x)

    x = Flatten()(x)

    for d in range(n_dense):
        units = max(64, dense_units // (2 ** d))
        x = Dense(units, activation="relu", kernel_regularizer=kernel_reg)(x)
        x = Dropout(dropout)(x)

    out_T = Dense(3, activation="softmax", name="T_output")(x)
    out_A = Dense(3, activation="softmax", name="A_output")(x)

    model = Model(inputs=inp, outputs=[out_T, out_A])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={
            "T_output": "categorical_crossentropy",
            "A_output": "categorical_crossentropy",
        },
        metrics={
            "T_output": ["accuracy"],
            "A_output": ["accuracy"],
        }
    )

    return model
# =============================================================================
# OPTUNA OBJECTIVE - MULTI-OUTPUT: T + A
# =============================================================================
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


class ExactMatchPruningCallback(tf.keras.callbacks.Callback):
    """
    Berechnet nach jeder Epoche den echten Exact Match auf dem Validation-Set
    (statt nur Accuracy-Mittelwert) und meldet ihn an Optuna.
    Sobald Optuna den Trial gegenüber dem Median schlechter findet -> Abbruch.
    """
    def __init__(self, trial, fold, X_val, yT_val, yA_val):
        super().__init__()
        self.trial = trial
        self.fold = fold
        self.X_val = X_val
        self.true_T = np.argmax(yT_val, axis=1)
        self.true_A = np.argmax(yA_val, axis=1)

    def on_epoch_end(self, epoch, logs=None):
        pred_T_prob, pred_A_prob = self.model.predict(self.X_val, verbose=0)
        pred_T = np.argmax(pred_T_prob, axis=1)
        pred_A = np.argmax(pred_A_prob, axis=1)
        exact_match = float(np.mean((pred_T == self.true_T) & (pred_A == self.true_A)))

        step = (self.fold - 1) * 1000 + epoch  # eindeutiger Step über Folds
        self.trial.report(exact_match, step)

        print(
            f"  [Trial {self.trial.number} | Fold {self.fold} | Epoch {epoch+1}] "
            f"ExactMatch={exact_match:.4f}",
            flush=True
        )

        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.TrialPruned(
                f"Pruned at trial {self.trial.number}, fold {self.fold}, epoch {epoch+1}"
            )


def optuna_objective_factory(X_trainval, yT_trainval, yA_trainval, y_trainval_int):

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs     = trial.suggest_int("epochs", 15, 50, step=5)

        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        fold_exact_matches = []
        fold_T_accs = []
        fold_A_accs = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_trainval, y_trainval_int), 1):

            X_tr_full = X_trainval[tr_idx]
            yT_tr_full = yT_trainval[tr_idx]
            yA_tr_full = yA_trainval[tr_idx]

            n_tr = len(X_tr_full)
            val_n = max(1, int(0.15 * n_tr))

            rng = np.random.default_rng(1000 + fold)
            perm = rng.permutation(n_tr)

            val_local = perm[:val_n]
            tr_local  = perm[val_n:]

            X_tr = X_tr_full[tr_local]
            yT_tr = yT_tr_full[tr_local]
            yA_tr = yA_tr_full[tr_local]

            X_val = X_tr_full[val_local]
            yT_val = yT_tr_full[val_local]
            yA_val = yA_tr_full[val_local]

            model = build_model_optuna(trial, input_shape=X_trainval.shape[1:])

            es = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
                restore_best_weights=True,
                verbose=0
            )

            lr_sched = ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=0
            )

            pruning_cb = ExactMatchPruningCallback(trial, fold, X_val, yT_val, yA_val)

            model.fit(
                X_tr,
                {
                    "T_output": yT_tr,
                    "A_output": yA_tr,
                },
                validation_data=(
                    X_val,
                    {
                        "T_output": yT_val,
                        "A_output": yA_val,
                    }
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es, lr_sched, pruning_cb],
                verbose=0
            )

            pred_T_prob, pred_A_prob = model.predict(X_val, verbose=0)

            pred_T = np.argmax(pred_T_prob, axis=1)
            pred_A = np.argmax(pred_A_prob, axis=1)

            true_T = np.argmax(yT_val, axis=1)
            true_A = np.argmax(yA_val, axis=1)

            T_acc = accuracy_score(true_T, pred_T)
            A_acc = accuracy_score(true_A, pred_A)

            exact_match = np.mean((pred_T == true_T) & (pred_A == true_A))

            fold_T_accs.append(float(T_acc))
            fold_A_accs.append(float(A_acc))
            fold_exact_matches.append(float(exact_match))

            del model
            tf.keras.backend.clear_session()
            gc.collect()

        mean_T_acc = float(np.mean(fold_T_accs))
        mean_A_acc = float(np.mean(fold_A_accs))
        mean_exact_match = float(np.mean(fold_exact_matches))

        trial.set_user_attr("mean_T_accuracy", mean_T_acc)
        trial.set_user_attr("mean_A_accuracy", mean_A_acc)
        trial.set_user_attr("mean_exact_match", mean_exact_match)

        return mean_exact_match

    return objective
# =============================================================================
# STUDIE MIT SQLITE-STORAGE (CRASH-SICHER!)
# =============================================================================
sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
# Konservativ: erst nach 20 Epochen vergleichen, erst nachdem 20 Trials komplett
# durchgelaufen sind ueberhaupt prunen. So gehen Spaet-Konvergierer nicht verloren.
pruner  = optuna.pruners.MedianPruner(n_warmup_steps=20, n_startup_trials=20)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
    storage="sqlite:///optuna_TA_MultiOutput.db",
    study_name="bayesian_TA_MultiOutput",
    load_if_exists=True
)

# --- FIX: Nur abgeschlossene Trials zählen, nicht alle ---
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if len(study.trials) > 0:
    logging.info(f"Fortsetzen: {len(study.trials)} Trials vorhanden ({len(completed_trials)} abgeschlossen)")
    if completed_trials:
        logging.info(f"Bisheriger Bester: {study.best_value:.6f}")
    else:
        logging.info("Noch kein abgeschlossener Trial mit Wert – starte neu.")

remaining_trials = 200 - len(study.trials)
if remaining_trials > 0:
    objective = optuna_objective_factory(X_trainval, yT_trainval, yA_trainval, y_trainval_int)
    study.optimize(objective, n_trials=remaining_trials, catch=(Exception,))

# --- FIX: best_value / best_trial nur abrufen wenn completed Trials existieren ---
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("\n===== Optuna Ergebnis (Exact Match) =====")
if completed_trials:
    print("Best Exact Match:", study.best_value)
    print("Best Params:", study.best_trial.params)
else:
    print("Keine abgeschlossenen Trials – Optimierung fehlgeschlagen oder alle Trials gepruned.")
    raise RuntimeError("Keine abgeschlossenen Trials vorhanden. Bitte Optuna-DB prüfen oder neu starten.")

# =============================================================================
# FINALES MODELL
# =============================================================================
def train_final_with_best(best_params, X_tv, yT_tv, yA_tv):
    fixed_trial = optuna.trial.FixedTrial(best_params)
    model = build_model_optuna(fixed_trial, input_shape=X_tv.shape[1:])

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    lr_sched = ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_tv,
        {
            "T_output": yT_tv,
            "A_output": yA_tv,
        },
        validation_split=0.1,
        epochs=best_params.get("epochs", 30),
        batch_size=best_params.get("batch_size", 16),
        callbacks=[es, lr_sched],
        verbose=1
    )

    return model, history


final_model, final_history = train_final_with_best(
    study.best_trial.params,
    X_trainval,
    yT_trainval,
    yA_trainval
)

print("\n===== Holdout-Test (X_test) =====")

eval_results = final_model.evaluate(
    X_test,
    {
        "T_output": yT_test,
        "A_output": yA_test,
    },
    verbose=0
)

print("Evaluate Results:", eval_results)

pred_T_prob, pred_A_prob = final_model.predict(X_test, batch_size=32, verbose=0)

pred_T = np.argmax(pred_T_prob, axis=1)
pred_A = np.argmax(pred_A_prob, axis=1)

true_T = np.argmax(yT_test, axis=1)
true_A = np.argmax(yA_test, axis=1)

T_acc = accuracy_score(true_T, pred_T)
A_acc = accuracy_score(true_A, pred_A)
exact_match_test = np.mean((pred_T == true_T) & (pred_A == true_A))

print(f"T-Accuracy: {T_acc:.4f} ({T_acc*100:.2f}%)")
print(f"A-Accuracy: {A_acc:.4f} ({A_acc*100:.2f}%)")
print(f"Combined Exact Match: {exact_match_test:.4f} ({exact_match_test*100:.2f}%)")

print("\n=== Classification Report Tiefziehen T1 bis T3 ===")
print(classification_report(true_T, pred_T, target_names=["T1", "T2", "T3"]))

print("\n=== Classification Report Abstrecken A1 bis A3 ===")
print(classification_report(true_A, pred_A, target_names=["A1", "A2", "A3"]))

# --- Confusion Matrix T ---
cm_T = confusion_matrix(true_T, pred_T)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_T,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["T1", "T2", "T3"],
    yticklabels=["T1", "T2", "T3"]
)
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Konfusionsmatrix Tiefziehen")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Confusion_Matrix_T.png"))
plt.show()


# --- Confusion Matrix A ---
cm_A = confusion_matrix(true_A, pred_A)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_A,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["A1", "A2", "A3"],
    yticklabels=["A1", "A2", "A3"]
)
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Konfusionsmatrix Abstrecken")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Confusion_Matrix_A.png"))
plt.show()
try:
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "optuna_trials_TA_MultiOutput.csv"), index=False)
    print("Trials gespeichert: optuna_trials_TA_MultiOutput.csv")
except Exception as e:
    print("Trials-Speichern fehlgeschlagen:", e)

final_model.save(os.path.join(output_dir, "final_model_TA_MultiOutput.keras"))
print("\n✅ Modell gespeichert: final_model_TA_MultiOutput.keras")
print("✅ FERTIG!")
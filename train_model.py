"""
train_model.py - train the sign SEQUENCE classifier and export it for desktop + mobile.

    python train_model.py                 # normal run
    python train_model.py --group-by person   # strictest test: unseen PEOPLE

Pipeline
    1. load clips/<label>/*.npy
    2. split into train/test BY SESSION (or person) so near-identical clips never leak
    3. augment the training clips only
    4. train a 1D-CNN (pure TFLite built-ins -> runs on Android/iOS without extra ops)
    5. honest evaluation on the held-out sessions: accuracy, per-class report, confusion matrix
    6. retrain on ALL data for the best epoch count and export:
         sign_language_model.keras / .tflite / labels.json / model_meta.json
         reports/confusion_matrix.png, reports/training_curves.png, reports/metrics.json
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from features import (BASE_DIR, DATA_DIR, FRAME_DIM, MODEL_DIM, SEQ_LEN, WINDOW_SECONDS,
                      add_velocity, augment_clip, load_dataset)

REPORT_DIR = os.path.join(BASE_DIR, "reports")


# --------------------------------------------------------------------------- #
def split_indices(y, groups, test_size, seed):
    """Group-aware split; guarantees every class is in both train and test."""
    ug = np.unique(groups)
    if len(ug) >= 3:
        for s in range(seed, seed + 300):
            tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(y, y, groups))
            if set(y[tr]) == set(y) and set(y[te]) == set(y):
                return tr, te, f"grouped ({len(np.unique(groups[te]))} held-out groups)"
        print("  [warn] could not find a group split containing every class in both sets")
    print("  [WARN] fewer than 3 recording sessions -> falling back to a random clip split.\n"
          "         Accuracy will be OPTIMISTIC. Record at least 3 sessions/people for an honest number.")
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    return tr, te, "random (optimistic)"


def augment_set(X, y, factor, rng):
    Xs, ys = [X], [y]
    for _ in range(factor):
        Xs.append(np.stack([augment_clip(c, rng) for c in X]))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def build_model(num_classes, adapt_data):
    L = tf.keras.layers
    norm = L.Normalization(axis=-1)
    norm.adapt(adapt_data)          # bakes mean/std into the exported model

    inp = L.Input(shape=(SEQ_LEN, MODEL_DIM))
    x = norm(inp)
    for filters, pool in ((64, True), (128, True), (128, False)):
        x = L.Conv1D(filters, 5, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.SpatialDropout1D(0.1)(x)
        if pool:
            x = L.MaxPooling1D(2)(x)
    x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
    x = L.Dropout(0.4)(x)
    x = L.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def fit(model, Xtr, ytr, Xva, yva, n_cls, epochs, patience, sample_weight):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=["accuracy"])
    monitor = "val_loss" if Xva is not None else "loss"
    cbs = [tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=6, min_lr=1e-5)]
    if Xva is not None:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,
                                                    restore_best_weights=True))
    val = (Xva, tf.keras.utils.to_categorical(yva, n_cls)) if Xva is not None else None
    return model.fit(Xtr, tf.keras.utils.to_categorical(ytr, n_cls), validation_data=val,
                     epochs=epochs, batch_size=32, sample_weight=sample_weight,
                     callbacks=cbs, verbose=2, shuffle=True)


def save_plots(history, cm, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(REPORT_DIR, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["loss"], label="train"); ax[0].plot(history["val_loss"], label="held-out")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history["accuracy"], label="train"); ax[1].plot(history["val_accuracy"], label="held-out")
    ax[1].set_title("Accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "training_curves.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 3, 1.0 * len(classes) + 2.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (held-out sessions)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150); plt.close(fig)


def export(model, classes, Xcheck):
    model.save(os.path.join(BASE_DIR, "sign_language_model.keras"))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    path = os.path.join(BASE_DIR, "sign_language_model.tflite")
    with open(path, "wb") as f:
        f.write(tfl)

    with open(os.path.join(BASE_DIR, "labels.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(BASE_DIR, "model_meta.json"), "w") as f:
        json.dump({"seq_len": SEQ_LEN, "window_seconds": WINDOW_SECONDS,
                   "frame_dim": FRAME_DIM, "model_dim": MODEL_DIM, "classes": classes}, f, indent=2)

    # parity check: TFLite must agree with Keras
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    n = min(len(Xcheck), 100)
    keras_p = model.predict(Xcheck[:n], verbose=0)
    agree, maxdiff = 0, 0.0
    for i in range(n):
        interp.set_tensor(inp["index"], Xcheck[i:i + 1].astype(np.float32))
        interp.invoke()
        p = interp.get_tensor(out["index"])[0]
        agree += int(p.argmax() == keras_p[i].argmax())
        maxdiff = max(maxdiff, float(np.abs(p - keras_p[i]).max()))
    print(f"TFLite vs Keras: {agree}/{n} identical predictions, max prob diff {maxdiff:.5f}  "
          f"({os.path.getsize(path) / 1024:.0f} KB)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--group-by", choices=["session", "person"], default="session")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--aug", type=int, default=12, help="augmented copies per training clip")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-final", action="store_true", help="skip the retrain-on-all-data stage")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    X, y_str, groups = load_dataset(args.data_dir, args.group_by)
    if len(X) == 0:
        raise SystemExit(f"No clips found in {args.data_dir}. Record some with tracker.py first.")
    classes = sorted(set(y_str))
    if len(classes) < 2:
        raise SystemExit("Need at least 2 classes (include a 'none' class!).")
    y = np.array([classes.index(c) for c in y_str])
    n_cls = len(classes)

    print(f"\nDataset: {len(X)} clips, {len(np.unique(groups))} groups ({args.group_by}s)")
    for i, c in enumerate(classes):
        print(f"  {c:<14} {int((y == i).sum()):>4} clips")
    if "none" not in classes:
        print("  [warn] no 'none' class: the model will force every movement into a word.")
    small = [c for i, c in enumerate(classes) if (y == i).sum() < 30]
    if small:
        print(f"  [warn] under 30 clips for {small} - record more (aim for 60+).")

    # ---- Stage 1: honest evaluation ----
    tr, te, how = split_indices(y, groups, args.test_size, args.seed)
    print(f"\nSplit: {how}: {len(tr)} train clips / {len(te)} test clips")
    Xtr, ytr = augment_set(X[tr], y[tr], args.aug, rng)
    Xtr_m, Xte_m = add_velocity(Xtr), add_velocity(X[te])

    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=ytr)
    model = build_model(n_cls, Xtr_m)
    print(f"\nStage 1: training on {len(Xtr_m)} (augmented) clips, evaluating on held-out data")
    hist = fit(model, Xtr_m, ytr, Xte_m, y[te], n_cls, args.epochs, 20, cw[ytr]).history
    best_epoch = int(np.argmin(hist["val_loss"])) + 1

    probs = model.predict(Xte_m, verbose=0)
    pred = probs.argmax(1)
    acc = float((pred == y[te]).mean())
    cm = confusion_matrix(y[te], pred, labels=range(n_cls))
    print(f"\n=== HELD-OUT ACCURACY: {acc * 100:.1f}%  (best epoch {best_epoch}) ===")
    print(classification_report(y[te], pred, labels=range(n_cls), target_names=classes, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):\n", cm)
    conf = probs.max(1)
    if (pred == y[te]).any():
        print(f"Mean confidence when correct: {conf[pred == y[te]].mean():.2f}")
    if (pred != y[te]).any():
        print(f"Mean confidence when WRONG:   {conf[pred != y[te]].mean():.2f}")
    save_plots(hist, cm, classes)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump({"held_out_accuracy": acc, "split": how, "classes": classes,
                   "confusion_matrix": cm.tolist(), "clips": int(len(X)),
                   "best_epoch": best_epoch}, f, indent=2)

    # ---- Stage 2: retrain on everything, export ----
    if args.no_final:
        final_model, check = model, Xte_m
    else:
        print("\nStage 2: retraining on ALL data for the best epoch count, then exporting")
        Xall, yall = augment_set(X, y, args.aug, rng)
        Xall_m = add_velocity(Xall)
        cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=yall)
        final_model = build_model(n_cls, Xall_m)
        fit(final_model, Xall_m, yall, None, None, n_cls, max(15, int(best_epoch * 1.15)), 0, cw[yall])
        check = add_velocity(X)
    export(final_model, classes, check)
    print("\nDONE. Exported sign_language_model.tflite, .keras, labels.json, model_meta.json")
    print("Reports saved in ./reports  (put confusion_matrix.png in your slides).")


if __name__ == "__main__":
    main()

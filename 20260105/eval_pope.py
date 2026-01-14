# eval_pope.py
import argparse
import os
import json
from datetime import datetime
from typing import Any, Dict, List

from PIL import Image
from tqdm import tqdm

from orbit_lib import OrbitRunner, parse_layers, norm_yesno, set_seed

def load_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    # jsonl
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def get_question(ex: Dict[str, Any]) -> str:
    for k in ["question", "query", "text"]:
        if k in ex:
            return str(ex[k])
    raise KeyError("No question field")

def get_label(ex: Dict[str, Any]) -> str:
    for k in ["label", "answer", "gt", "gt_answer", "gold"]:
        if k in ex:
            v = ex[k]
            if isinstance(v, str):
                return norm_yesno(v)
            if isinstance(v, (int, float)):
                return "yes" if int(v) == 1 else "no"
    raise KeyError("No label field")

def resolve_image_path(ex: Dict[str, Any], image_root: str) -> str:
    for k in ["image", "image_path", "img", "img_path", "file_name"]:
        if k in ex:
            p = str(ex[k])
            if os.path.isabs(p) and os.path.exists(p):
                return p
            p2 = os.path.join(image_root, p)
            if os.path.exists(p2):
                return p2

    if "image_id" in ex:
        iid = int(ex["image_id"])
        cand = os.path.join(image_root, f"COCO_val2014_{iid:012d}.jpg")
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(image_root, f"{iid:012d}.jpg")
        if os.path.exists(cand2):
            return cand2
        cand3 = os.path.join(image_root, f"{iid}.jpg")
        if os.path.exists(cand3):
            return cand3

    raise FileNotFoundError("Cannot resolve image path for this example")

def metrics(preds: List[str], labels: List[str]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for p, y in zip(preds, labels):
        if y == "yes":
            if p == "yes": tp += 1
            else: fn += 1
        else:
            if p == "yes": fp += 1
            else: tn += 1
    n = tp + tn + fp + fn
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    yes_ratio = (tp + fp) / max(n, 1)
    return dict(acc=acc, precision=prec, recall=rec, f1=f1, yes_ratio=yes_ratio, tp=tp, tn=tn, fp=fp, fn=fn, n=n)

def append_run_summary(out_dir: str, run_obj: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary_runs.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                old = json.load(f)
            except Exception:
                old = []
        if isinstance(old, dict):
            runs = old.get("runs", [])
        else:
            runs = old
    else:
        runs = []

    runs.append(run_obj)
    payload = {"runs": runs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def eval_split(
    runner: OrbitRunner,
    name: str,
    data: List[Dict[str, Any]],
    image_root: str,
    out_dir: str,
    layer_indices: List[int],
    beta1: float,
    beta2: float,
    tau_scale: float,
    max_samples: int = 0,
    use_veto: bool = False,
    veto_delta: float = 0.0,
    logit_only: bool = False,
    use_cache: bool = True,
):
    if max_samples > 0:
        data = data[:max_samples]

    def run(enable_orbit: bool):
        preds, labels = [], []
        rows = []
        tag = "orbit" if enable_orbit else "base"

        for ex in tqdm(data, desc=f"{name}-{tag}"):
            q = get_question(ex)
            y = get_label(ex)
            img_path = resolve_image_path(ex, image_root)
            img = Image.open(img_path).convert("RGB")

            # logit_only: do_yesno=True 且 max_new_tokens=0
            max_new_tokens = 0 if logit_only else 4

            ans, info = runner.generate(
                image=img,
                question=q,
                enable_orbit=enable_orbit,
                layer_indices=layer_indices,
                beta1=beta1,
                beta2=beta2,
                tau_scale=tau_scale,
                max_new_tokens=max_new_tokens,
                do_yesno=True,
                use_cache=use_cache,
                apply_decode=True,
                debug_once=False,
                return_info=True,
                logit_only=logit_only,
                use_veto=use_veto,
                veto_delta=veto_delta,
            )

            p = norm_yesno(ans)
            preds.append(p); labels.append(y)

            row = {
                "image": img_path,
                "question": q,
                "label": y,
                "pred": p,
                "raw": ans,
            }
            # 保存关键信息（veto / logit分数），便于论文写 ablation
            row["info"] = {
                "family": info.get("family"),
                "yesno_score_real": info.get("yesno_score_real"),
                "yesno_score_null": info.get("yesno_score_null"),
                "veto_gap": info.get("veto_gap"),
                "vetoed": info.get("vetoed"),
                "yesno_mode": (info.get("yesno_extra_real") or {}).get("mode"),
            }
            rows.append(row)

        met = metrics(preds, labels)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}_{tag}_preds.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, f"{name}_{tag}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(met, f, ensure_ascii=False, indent=2)
        return met

    m_base = run(False)
    m_orbit = run(True)

    print(f"\n=== {name} ===")
    print("BASE :", {k: round(m_base[k], 4) for k in ["acc", "precision", "recall", "f1", "yes_ratio"]})
    print("ORBIT:", {k: round(m_orbit[k], 4) for k in ["acc", "precision", "recall", "f1", "yes_ratio"]})
    return {"base": m_base, "orbit": m_orbit}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--bank_path", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--pope_jsons", type=str, nargs="+", required=True)
    ap.add_argument("--layers", type=str, default="-9")
    ap.add_argument("--beta1", type=float, default=10.0)
    ap.add_argument("--beta2", type=float, default=0.0)
    ap.add_argument("--tau_scale", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="results_pope")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_cache", action="store_true")

    # ✅ NEW(1): veto 参数（两个参数）
    ap.add_argument("--use_veto", action="store_true")
    ap.add_argument("--veto_delta", type=float, default=0.0)

    # ✅ NEW(2): logit_only（do_yesno 时不调用 generate）
    ap.add_argument("--logit_only", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    runner = OrbitRunner(args.model_path, bank_path=args.bank_path)
    layer_indices = parse_layers(args.layers, runner.num_layers)

    summary = {}
    for jp in args.pope_jsons:
        name = os.path.splitext(os.path.basename(jp))[0]
        data = load_json_any(jp)
        summary[name] = eval_split(
            runner=runner,
            name=name,
            data=data,
            image_root=args.image_root,
            out_dir=args.out_dir,
            layer_indices=layer_indices,
            beta1=args.beta1,
            beta2=args.beta2,
            tau_scale=args.tau_scale,
            max_samples=args.max_samples,
            use_veto=args.use_veto,
            veto_delta=args.veto_delta,
            logit_only=args.logit_only,
            use_cache=args.use_cache,
        )

    # 保存一次性 summary（当次）
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ✅ 续写到全局 runs
    run_obj = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cfg": {
            "model_path": args.model_path,
            "bank_path": args.bank_path,
            "image_root": args.image_root,
            "layers": args.layers,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "tau_scale": args.tau_scale,
            "use_veto": args.use_veto,
            "veto_delta": args.veto_delta,
            "logit_only": args.logit_only,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "use_cache": args.use_cache,
        },
        "results": summary,
    }
    path = append_run_summary(args.out_dir, run_obj)
    print(f"🧾 Appended run summary -> {path}")

if __name__ == "__main__":
    main()

# grid_search_pope.py
import argparse
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

from orbit_lib import OrbitRunner, parse_layers, norm_yesno, set_seed

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if txt.startswith("["):
        return json.loads(txt)
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def resolve_image_path(ex: Dict[str, Any], image_root: str) -> str:
    if "image_id" in ex:
        iid = int(ex["image_id"])
        cand = os.path.join(image_root, f"COCO_val2014_{iid:012d}.jpg")
        if os.path.exists(cand):
            return cand
    for k in ["image", "image_path", "img", "img_path", "file_name"]:
        if k in ex:
            p = str(ex[k])
            if os.path.isabs(p) and os.path.exists(p):
                return p
            p2 = os.path.join(image_root, p)
            if os.path.exists(p2):
                return p2
    raise FileNotFoundError("Cannot resolve image path")

def get_q(ex: Dict[str, Any]) -> str:
    return str(ex.get("question") or ex.get("query") or ex.get("text"))

def get_y(ex: Dict[str, Any]) -> str:
    v = ex.get("label", ex.get("answer", ex.get("gt", ex.get("gold"))))
    if isinstance(v, str):
        return norm_yesno(v)
    if isinstance(v, (int, float)):
        return "yes" if int(v) == 1 else "no"
    return "no"

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

def parse_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def eval_one(
    runner: OrbitRunner,
    data: List[Dict[str, Any]],
    image_root: str,
    layer_indices: List[int],
    enable_orbit: bool,
    beta1: float,
    beta2: float,
    tau_scale: float,
    use_veto: bool,
    veto_delta: float,
    max_samples: int,
    use_cache: bool,
) -> Dict[str, float]:
    if max_samples > 0:
        data = data[:max_samples]

    preds, labels = [], []
    for ex in data:
        q = get_q(ex)
        y = get_y(ex)
        img_path = resolve_image_path(ex, image_root)
        img = Image.open(img_path).convert("RGB")

        ans = runner.generate(
            image=img,
            question=q,
            enable_orbit=enable_orbit,
            layer_indices=layer_indices,
            beta1=beta1,
            beta2=beta2,
            tau_scale=tau_scale,
            max_new_tokens=0,     # logit-only
            do_yesno=True,
            use_cache=use_cache,
            apply_decode=True,
            debug_once=False,
            return_info=False,
            logit_only=True,
            use_veto=use_veto,
            veto_delta=veto_delta,
        )
        preds.append(norm_yesno(ans))
        labels.append(y)

    return metrics(preds, labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--bank_path", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--adversarial_json", type=str, required=True)
    ap.add_argument("--other_jsons", type=str, nargs="*", default=[])
    ap.add_argument("--layers", type=str, default="-9")
    ap.add_argument("--beta2", type=float, default=0.0)
    ap.add_argument("--beta1_list", type=str, required=True)
    ap.add_argument("--tau_list", type=str, required=True)
    ap.add_argument("--veto_list", type=str, required=True)
    ap.add_argument("--also_no_veto", action="store_true")
    ap.add_argument("--prec_drop", type=float, default=0.0, help="precision allowed to drop from baseline")
    ap.add_argument("--prec_min", type=float, default=0.0, help="absolute minimum precision constraint")
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_cache", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    runner = OrbitRunner(args.model_path, bank_path=args.bank_path)
    layer_indices = parse_layers(args.layers, runner.num_layers)

    adv = load_json(args.adversarial_json)
    others = [(os.path.splitext(os.path.basename(p))[0], load_json(p)) for p in args.other_jsons]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = os.path.join(args.out_dir, run_id)
    os.makedirs(out_run, exist_ok=True)

    print("\n========================")
    print("1) Baseline on adversarial (logit-only)")
    print("========================\n")
    base_met = eval_one(
        runner, adv, args.image_root, layer_indices,
        enable_orbit=False,
        beta1=0.0, beta2=args.beta2, tau_scale=1.0,
        use_veto=False, veto_delta=0.0,
        max_samples=args.max_samples,
        use_cache=args.use_cache,
    )
    print("BASE:", {k: round(base_met[k], 4) for k in ["acc", "precision", "recall", "f1", "yes_ratio"]})

    with open(os.path.join(out_run, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(base_met, f, ensure_ascii=False, indent=2)

    beta1_list = parse_list(args.beta1_list)
    tau_list = parse_list(args.tau_list)
    veto_list = parse_list(args.veto_list)

    # build candidate configs
    cfgs: List[Dict[str, Any]] = []
    for b1 in beta1_list:
        for tau in tau_list:
            if args.also_no_veto:
                cfgs.append({"beta1": b1, "tau_scale": tau, "use_veto": False, "veto_delta": 0.0})
            for vd in veto_list:
                cfgs.append({"beta1": b1, "tau_scale": tau, "use_veto": True, "veto_delta": vd})

    print("\n========================")
    print(f"2) Grid search candidates = {len(cfgs)}")
    print("========================")

    # precision constraint
    prec_threshold = max(args.prec_min, float(base_met["precision"]) - args.prec_drop)

    results = []
    for cfg in tqdm(cfgs, desc="grid"):
        met = eval_one(
            runner, adv, args.image_root, layer_indices,
            enable_orbit=True,
            beta1=float(cfg["beta1"]),
            beta2=float(args.beta2),
            tau_scale=float(cfg["tau_scale"]),
            use_veto=bool(cfg["use_veto"]),
            veto_delta=float(cfg["veto_delta"]),
            max_samples=args.max_samples,
            use_cache=args.use_cache,
        )
        results.append({"cfg": cfg, "met": met})

        with open(os.path.join(out_run, "grid_results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"cfg": cfg, "met": met}, ensure_ascii=False) + "\n")

    # select best by F1 under precision constraint
    feasible = [r for r in results if r["met"]["precision"] >= prec_threshold]
    if not feasible:
        feasible = results  # fall back

    feasible.sort(key=lambda r: (r["met"]["f1"], r["met"]["acc"]), reverse=True)
    best = feasible[0]
    topk = feasible[: max(1, args.topk)]

    print("\n========================")
    print("3) Best config (by F1 with precision constraint)")
    print("========================")
    print("BEST cfg:", best["cfg"])
    print("BEST met:", {k: round(best["met"][k], 4) for k in ["acc", "precision", "recall", "f1", "yes_ratio"]})

    with open(os.path.join(out_run, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_run, "topk.json"), "w", encoding="utf-8") as f:
        json.dump(topk, f, ensure_ascii=False, indent=2)

    # evaluate best on other splits
    summary = {
        "run_id": run_id,
        "baseline_adv": base_met,
        "prec_threshold": prec_threshold,
        "best": best,
        "others": {},
        "cfg_meta": {
            "model_path": args.model_path,
            "bank_path": args.bank_path,
            "image_root": args.image_root,
            "layers": args.layers,
            "beta2": args.beta2,
            "beta1_list": beta1_list,
            "tau_list": tau_list,
            "veto_list": veto_list,
            "also_no_veto": args.also_no_veto,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "use_cache": args.use_cache,
        }
    }

    for name, data in others:
        met0 = eval_one(
            runner, data, args.image_root, layer_indices,
            enable_orbit=False,
            beta1=0.0, beta2=args.beta2, tau_scale=1.0,
            use_veto=False, veto_delta=0.0,
            max_samples=args.max_samples,
            use_cache=args.use_cache,
        )
        met1 = eval_one(
            runner, data, args.image_root, layer_indices,
            enable_orbit=True,
            beta1=float(best["cfg"]["beta1"]),
            beta2=float(args.beta2),
            tau_scale=float(best["cfg"]["tau_scale"]),
            use_veto=bool(best["cfg"]["use_veto"]),
            veto_delta=float(best["cfg"]["veto_delta"]),
            max_samples=args.max_samples,
            use_cache=args.use_cache,
        )
        summary["others"][name] = {"base": met0, "best": met1}

    with open(os.path.join(out_run, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Grid search done. Saved to: {out_run}")
    print("   - baseline_metrics.json")
    print("   - grid_results.jsonl")
    print("   - topk.json / best.json")
    print("   - summary.json")

if __name__ == "__main__":
    main()

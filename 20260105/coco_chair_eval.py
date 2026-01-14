import argparse
import os
import json
import re
from typing import Dict, List, Set, Tuple
from PIL import Image
from tqdm import tqdm

from orbit_lib import generate_answer, load_llava, parse_layers, set_seed

def load_instances(instances_json: str) -> Tuple[Dict[int, Set[str]], Dict[int, str]]:
    """
    Build image_id -> set(category_name) from COCO instances json.
    Also return image_id -> file_name if present.
    """
    with open(instances_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}
    img_id_to_file = {im["id"]: im.get("file_name", "") for im in coco.get("images", [])}

    img_objs: Dict[int, Set[str]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        name = cats.get(cat_id, None)
        if name is None:
            continue
        img_objs.setdefault(img_id, set()).add(name)
    return img_objs, img_id_to_file

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())

def find_mentions(caption: str, obj_names: List[str]) -> List[str]:
    """
    Approximate object mention detection:
    - match multi-word phrases by substring with word boundaries
    - match single words by token match + simple plural stripping
    """
    cap = caption.lower()
    toks = tokenize(cap)
    tokset = set(toks)

    mentions = []
    # multi-word first
    for obj in obj_names:
        if " " in obj:
            pattern = r"\b" + re.escape(obj) + r"\b"
            if re.search(pattern, cap):
                mentions.append(obj)
        else:
            # singular/plural heuristic
            if obj in tokset:
                mentions.append(obj)
            elif obj.endswith("s") and obj[:-1] in tokset:
                mentions.append(obj)
            elif (obj + "s") in tokset:
                mentions.append(obj)
    return mentions

def chair_metrics(captions: List[Dict], img_objs: Dict[int, Set[str]], obj_names: List[str]) -> Dict[str, float]:
    """
    CHAIR-s: fraction of sentences with >=1 hallucinated object
    CHAIR-i: fraction of hallucinated object mentions among all object mentions
    (This follows the core idea; exact official CHAIR uses richer synonyms.)
    """
    sent_hall = 0
    total_sent = 0
    hall_inst = 0
    total_inst = 0

    for ex in captions:
        img_id = int(ex["image_id"])
        cap = ex["caption"]
        present = img_objs.get(img_id, set())

        mentions = find_mentions(cap, obj_names)
        total_sent += 1

        if len(mentions) == 0:
            continue

        total_inst += len(mentions)
        hallucinated = [m for m in mentions if m not in present]
        hall_inst += len(hallucinated)
        if len(hallucinated) > 0:
            sent_hall += 1

    chair_s = sent_hall / max(total_sent, 1)
    chair_i = hall_inst / max(total_inst, 1)
    return {
        "CHAIR_s": chair_s,
        "CHAIR_i": chair_i,
        "sent_hall": sent_hall,
        "total_sent": total_sent,
        "hall_inst": hall_inst,
        "total_inst": total_inst,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--bank_path", type=str, required=True)
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--instances_json", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=1000)
    ap.add_argument("--layers", type=str, default="-9")
    ap.add_argument("--beta1", type=float, default=10.0)
    ap.add_argument("--beta2", type=float, default=6.0)
    ap.add_argument("--tau_scale", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="results_chair")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # layer mapping
    model, _, _, _ = load_llava(args.model_path)
    num_layers = len(model.language_model.model.layers)
    layer_indices = parse_layers(args.layers, num_layers)

    img_objs, img_id_to_file = load_instances(args.instances_json)
    # object vocabulary from categories
    obj_names = sorted({name for s in img_objs.values() for name in s})

    # pick image ids that exist in instances
    img_ids = list(img_objs.keys())
    img_ids = img_ids[: args.num_images]

    os.makedirs(args.out_dir, exist_ok=True)

    def gen_caps(enable_orbit: bool) -> List[Dict]:
        out = []
        for iid in tqdm(img_ids, desc=f"caption ({'ORBIT' if enable_orbit else 'BASE'})"):
            fn = img_id_to_file.get(iid, f"{iid:012d}.jpg")
            path = os.path.join(args.images_dir, fn)
            if not os.path.exists(path):
                # fallback: try raw id name
                path2 = os.path.join(args.images_dir, f"{iid:012d}.jpg")
                if os.path.exists(path2):
                    path = path2
                else:
                    continue
            img = Image.open(path).convert("RGB")
            cap = generate_answer(
                model_path=args.model_path,
                bank_path=args.bank_path,
                image=img,
                question="Describe the image in one sentence.",
                enable_orbit=enable_orbit,
                beta1=args.beta1,
                beta2=args.beta2,
                layer_indices=layer_indices,
                max_new_tokens=32,
                use_cache=False,
                debug_once=False,
                tau_scale=args.tau_scale,
                s_scale=1.0,
                min_gate=0.0,
                two_pass_veto=False,
            )
            out.append({"image_id": int(iid), "caption": cap.strip()})
        return out

    caps_base = gen_caps(enable_orbit=False)
    caps_orbit = gen_caps(enable_orbit=True)

    with open(os.path.join(args.out_dir, "captions_base.json"), "w", encoding="utf-8") as f:
        json.dump(caps_base, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "captions_orbit.json"), "w", encoding="utf-8") as f:
        json.dump(caps_orbit, f, ensure_ascii=False, indent=2)

    m_base = chair_metrics(caps_base, img_objs, obj_names)
    m_orbit = chair_metrics(caps_orbit, img_objs, obj_names)

    with open(os.path.join(args.out_dir, "chair_base.json"), "w", encoding="utf-8") as f:
        json.dump(m_base, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "chair_orbit.json"), "w", encoding="utf-8") as f:
        json.dump(m_orbit, f, ensure_ascii=False, indent=2)

    print("\n=== CHAIR (approx) ===")
    print("BASE :", {k: round(m_base[k], 4) for k in ["CHAIR_s","CHAIR_i"]})
    print("ORBIT:", {k: round(m_orbit[k], 4) for k in ["CHAIR_s","CHAIR_i"]})

if __name__ == "__main__":
    main()

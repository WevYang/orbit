# run_extraction.py
import argparse
import os
from typing import List
from orbit_lib import OrbitRunner, set_seed

def list_images(root: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--save_path", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=300)
    ap.add_argument("--prompts_per_family", type=int, default=80)
    ap.add_argument("--layers", type=str, default="-9")
    ap.add_argument("--k_prior", type=int, default=32)
    ap.add_argument("--l_evid", type=int, default=32)
    ap.add_argument("--tau_quantile", type=float, default=75.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    paths = list_images(args.image_root)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {args.image_root}")

    runner = OrbitRunner(args.model_path)
    runner.build_bank(
        image_paths=paths,
        save_path=args.save_path,
        num_images=args.num_images,
        prompts_per_family=args.prompts_per_family,
        layers=args.layers,
        k_prior=args.k_prior,
        l_evid=args.l_evid,
        tau_quantile=args.tau_quantile,
        batch_size=args.batch_size,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()

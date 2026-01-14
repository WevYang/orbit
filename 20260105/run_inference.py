import argparse
from PIL import Image
from orbit_lib import OrbitRunner, parse_layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--bank_path", type=str, required=True)
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--layers", type=str, default="-9")
    ap.add_argument("--beta1", type=float, default=10.0)
    ap.add_argument("--beta2", type=float, default=6.0)
    ap.add_argument("--tau_scale", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    runner = OrbitRunner(args.model_path, bank_path=args.bank_path)
    layer_indices = parse_layers(args.layers, runner.num_layers)
    img = Image.open(args.image_path).convert("RGB")

    print("\n" + "="*60)
    print("🛑 BASELINE")
    print("="*60)
    base = runner.generate(
        image=img, question=args.question,
        enable_orbit=False, layer_indices=layer_indices,
        max_new_tokens=64, use_cache=True
    )
    print("ASSISTANT:", base)

    print("\n" + "="*60)
    print("🛡️ ORBIT")
    print("="*60)
    ans = runner.generate(
        image=img, question=args.question,
        enable_orbit=True, layer_indices=layer_indices,
        beta1=args.beta1, beta2=args.beta2, tau_scale=args.tau_scale,
        max_new_tokens=64, use_cache=True, apply_decode=True,
        debug_once=args.debug
    )
    print("ASSISTANT:", ans)

if __name__ == "__main__":
    main()

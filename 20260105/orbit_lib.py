# orbit_lib.py
import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -----------------------------
# Prompt families (minimal + stable)
# -----------------------------
FAMILIES = ["existence", "attribute_color", "counting", "open_vqa", "caption"]

FAMILY_TEMPLATES: Dict[str, List[str]] = {
    "existence": [
        "Is there a {obj} in the image? Answer Yes or No.",
        "Do you see any {obj}? Answer Yes or No.",
        "Is {obj} present in this image? Answer Yes or No.",
    ],
    "attribute_color": [
        "What color is the {obj}?",
        "Tell me the color of the {obj}.",
        "What is the {obj}'s color?",
    ],
    "counting": [
        "How many {obj} are there in the image?",
        "Count the number of {obj} in the image.",
    ],
    "open_vqa": [
        "Answer the question based on the image: {q}",
        "Based on the image, {q}",
    ],
    "caption": [
        "Describe the image in one sentence.",
        "Write a short caption for this image.",
    ],
}

OBJ_POOL = ["person", "dog", "cat", "car", "hat", "pizza", "bicycle", "bus", "chair", "cup"]


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_null_image(size: int = 336) -> Image.Image:
    return Image.new("RGB", (size, size), (0, 0, 0))


def llava_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


def strip_assistant(decoded: str) -> str:
    if "ASSISTANT:" in decoded:
        return decoded.split("ASSISTANT:")[-1].strip()
    return decoded.strip()


def infer_family(question: str) -> str:
    q = (question or "").lower().strip()
    if "answer yes or no" in q or q.startswith(("is there", "are there", "do you see", "is ", "are ")):
        return "existence"
    if q.startswith("what color") or "color of" in q:
        return "attribute_color"
    if q.startswith("how many") or "count the number" in q:
        return "counting"
    if q.startswith(("describe", "caption", "write a short caption")):
        return "caption"
    return "open_vqa"


def norm_yesno(text: str) -> str:
    s = (text or "").strip().lower()
    if "yes" in s and "no" in s:
        return "yes" if s.find("yes") < s.find("no") else "no"
    if "yes" in s:
        return "yes"
    if "no" in s:
        return "no"
    return "no"


# -----------------------------
# Robust layer discovery (compat across versions)
# -----------------------------
def _find_longest_modulelist(root: nn.Module) -> nn.ModuleList:
    best: Optional[nn.ModuleList] = None
    best_len = -1

    def rec(m: nn.Module):
        nonlocal best, best_len
        for child in m.children():
            if isinstance(child, nn.ModuleList) and len(child) > best_len:
                best = child
                best_len = len(child)
            rec(child)

    rec(root)
    if best is None:
        raise RuntimeError("Could not find any nn.ModuleList (decoder layers) in the model.")
    return best


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Return the decoder layer ModuleList for LLaVA/LLaMA variants across versions.
    """
    lm = getattr(model, "language_model", None)
    if lm is not None:
        # common HF layout: LlamaForCausalLM has .model.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        # some versions: LlamaModel.layers
        if hasattr(lm, "layers"):
            return lm.layers
        # fallback search inside language_model
        return _find_longest_modulelist(lm)
    # fallback search in full model
    return _find_longest_modulelist(model)


def parse_layers(layers_str: str, num_layers: int) -> List[int]:
    """
    Support "-9,-13" or "23". Return decoder_idx list in [0, num_layers).
    """
    layers_str = (layers_str or "").strip()
    if not layers_str:
        return []
    out: List[int] = []
    for part in layers_str.split(","):
        part = part.strip()
        if not part:
            continue
        li = int(part)
        di = (num_layers + li) if li < 0 else li
        if not (0 <= di < num_layers):
            raise ValueError(f"Layer {li} -> decoder_idx {di} out of range (num_layers={num_layers})")
        out.append(di)
    return sorted(list(set(out)))


# -----------------------------
# PCA (CPU, stable for small N)
# -----------------------------
def pca_basis(X: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [N, D] float32 CPU
    return mean [D], U [D,k] orthonormal
    """
    X = X.float()
    mean = X.mean(dim=0, keepdim=True)  # [1,D]
    Xc = X - mean
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    U = Vh.transpose(0, 1)[:, :k].contiguous()  # [D,k]
    return mean.squeeze(0), U


# -----------------------------
# Bank structs
# -----------------------------
@dataclass
class OrbitLayerBank:
    mean_prior: torch.Tensor  # [D]
    U_prior: torch.Tensor     # [D,K]
    U_evid: torch.Tensor      # [D,L]
    tau: float
    s: float


@dataclass
class OrbitBank:
    families: Dict[str, Dict[int, OrbitLayerBank]]
    meta: Dict[str, Any]


def _bank_to_payload(bank: OrbitBank) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"meta": bank.meta, "families": {}}
    for fam, layer_map in bank.families.items():
        payload["families"][fam] = {}
        for di, b in layer_map.items():
            payload["families"][fam][str(di)] = {
                "mean_prior": b.mean_prior,
                "U_prior": b.U_prior,
                "U_evid": b.U_evid,
                "tau": float(b.tau),
                "s": float(b.s),
            }
    return payload


def _payload_to_bank(obj: Dict[str, Any]) -> OrbitBank:
    fams: Dict[str, Dict[int, OrbitLayerBank]] = {}
    families = obj.get("families", {})
    for fam, layer_map in families.items():
        fams[fam] = {}
        for k, v in layer_map.items():
            di = int(k)
            fams[fam][di] = OrbitLayerBank(
                mean_prior=v["mean_prior"].float().cpu(),
                U_prior=v["U_prior"].float().cpu(),
                U_evid=v["U_evid"].float().cpu(),
                tau=float(v["tau"]),
                s=float(v["s"]),
            )
    return OrbitBank(families=fams, meta=obj.get("meta", {}))


def save_bank(bank: OrbitBank, path: str) -> None:
    """
    Save as a pure dict+tensors payload to avoid torch 2.6+ weights_only issues.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = _bank_to_payload(bank)
    torch.save(payload, path)


def load_bank(path: str) -> OrbitBank:
    """
    Compatible with:
      - new safe dict payload
      - old pickled OrbitBank (fallback weights_only=False)
    """
    try:
        obj = torch.load(path, map_location="cpu")  # torch 2.6+ default weights_only=True
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, OrbitBank):
        # legacy
        return obj

    if isinstance(obj, dict) and "families" in obj:
        return _payload_to_bank(obj)

    raise ValueError(f"Unexpected bank format: {type(obj)}")


# -----------------------------
# Runner (loads model ONCE)
# -----------------------------
class OrbitRunner:
    def __init__(self, model_path: str, bank_path: Optional[str] = None):
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()

        self.layers = get_decoder_layers(self.model)
        self.num_layers = len(self.layers)
        print(f"🧱 Decoder layers = {self.num_layers}")

        self.bank: Optional[OrbitBank] = load_bank(bank_path) if bank_path else None
        self._handles: List[Any] = []
        self._family: str = "open_vqa"

        # device_map safe: push inputs to the first parameter device
        self.input_device = next(self.model.parameters()).device

    def close_orbit(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    # --------- hidden extraction (batched) ---------
    @torch.no_grad()
    def last_token_hidden_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        decoder_idx: int,
        batch_size: int = 4,
    ) -> torch.Tensor:
        assert len(images) == len(questions)
        outs: List[torch.Tensor] = []

        for i in range(0, len(images), batch_size):
            b_imgs = images[i:i + batch_size]
            b_qs = [llava_prompt(q) for q in questions[i:i + batch_size]]

            inputs = self.processor(text=b_qs, images=b_imgs, padding=True, return_tensors="pt")
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if k == "pixel_values":
                        inputs[k] = v.to(device=self.input_device, dtype=self.model.dtype)
                    else:
                        inputs[k] = v.to(device=self.input_device)

            out = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            hs = out.hidden_states  # (emb, layer1..layerN)
            h = hs[decoder_idx + 1]  # [B,T,D]

            attn = inputs.get("attention_mask", None)
            if attn is None:
                pos = torch.full((h.shape[0],), h.shape[1] - 1, device=h.device, dtype=torch.long)
            else:
                pos = attn.sum(dim=1) - 1
                pos = torch.clamp(pos, min=0)

            b_idx = torch.arange(h.shape[0], device=h.device)
            vec = h[b_idx, pos, :].detach().float().cpu()  # [B,D]
            outs.append(vec)

        return torch.cat(outs, dim=0)  # [N,D]

    # --------- build bank ---------
    @torch.no_grad()
    def build_bank(
        self,
        image_paths: List[str],
        save_path: str,
        num_images: int = 300,
        prompts_per_family: int = 80,
        layers: str = "-9",
        k_prior: int = 32,
        l_evid: int = 32,
        tau_quantile: float = 75.0,
        batch_size: int = 4,
        seed: int = 0,
    ) -> OrbitBank:
        set_seed(seed)
        image_paths = list(image_paths)
        random.shuffle(image_paths)
        image_paths = image_paths[:num_images]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        null_img = make_null_image()

        decoder_idxs = parse_layers(layers, self.num_layers)
        if not decoder_idxs:
            raise ValueError("No layers provided.")

        families: Dict[str, Dict[int, OrbitLayerBank]] = {fam: {} for fam in FAMILIES}

        def sample_prompts(fam: str, n: int) -> List[str]:
            temps = FAMILY_TEMPLATES[fam]
            outp = []
            for _ in range(n):
                t = random.choice(temps)
                if "{obj}" in t:
                    outp.append(t.format(obj=random.choice(OBJ_POOL)))
                elif "{q}" in t:
                    outp.append(t.format(q="What is happening in the image?"))
                else:
                    outp.append(t)
            return outp

        for fam in FAMILIES:
            prompts = sample_prompts(fam, prompts_per_family)

            # Prior: prompts × 1 null image (batched by prompts)
            prior_imgs = [null_img] * len(prompts)
            prior_qs = prompts

            for di in decoder_idxs:
                print(f"[bank] fam={fam} decoder_idx={di} ...")

                Xp = self.last_token_hidden_batch(prior_imgs, prior_qs, di, batch_size=batch_size)  # [P,D]
                mean_p, U_p = pca_basis(Xp, k_prior)

                # Evidence: real images with random prompt
                evid_qs = [random.choice(prompts) for _ in range(len(images))]
                Xe = self.last_token_hidden_batch(images, evid_qs, di, batch_size=batch_size)  # [N,D]

                Xe_c = Xe - mean_p.unsqueeze(0)
                proj_p = (Xe_c @ U_p) @ U_p.T
                R = Xe_c - proj_p
                _, U_e = pca_basis(R, l_evid)

                # log ratio distribution
                ap = Xe_c @ U_p
                ae = R @ U_e
                Ep = (ap ** 2).sum(dim=-1)
                Ee = (ae ** 2).sum(dim=-1)
                logratio = torch.log((Ep + 1e-8) / (Ee + 1e-8))

                tau = float(torch.quantile(logratio, tau_quantile / 100.0).item())
                s = float(logratio.std(unbiased=False).item() + 1e-6)

                ortho = float(torch.norm(U_p.T @ U_e).item())

                families[fam][di] = OrbitLayerBank(
                    mean_prior=mean_p.cpu(),
                    U_prior=U_p.cpu(),
                    U_evid=U_e.cpu(),
                    tau=tau,
                    s=s,
                )

                print(f"    tau={tau:.4f} s={s:.4f} (q={tau_quantile}%) ||Up^T Ue||={ortho:.6f}")

        bank = OrbitBank(
            families=families,
            meta=dict(
                model_path=self.model_path,
                num_images=num_images,
                prompts_per_family=prompts_per_family,
                layers=layers,
                k_prior=k_prior,
                l_evid=l_evid,
                tau_quantile=tau_quantile,
                batch_size=batch_size,
                seed=seed,
            ),
        )
        save_bank(bank, save_path)
        print(f"🎉 ORBIT bank saved to: {save_path}")
        self.bank = bank
        return bank

    # --------- ORBIT hook ---------
    def _make_hook(
        self,
        decoder_idx: int,
        beta1: float,
        beta2: float,
        tau_scale: float,
        apply_decode: bool,
        debug_once: bool,
    ):
        did_debug = {"v": False}

        def hook(module, inputs, output):
            if self.bank is None:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            if (not torch.is_tensor(hidden)) or hidden.ndim != 3 or hidden.shape[1] == 0:
                return output

            # decode step has seq_len==1 if use_cache=True
            if (hidden.shape[1] == 1) and (not apply_decode):
                return output

            fam = self._family
            layer_bank = self.bank.families.get(fam, {}).get(decoder_idx, None)
            if layer_bank is None:
                return output

            h_last = hidden[:, -1, :]  # [B,D]
            device = h_last.device
            dtype = h_last.dtype

            mean_p = layer_bank.mean_prior.to(device=device, dtype=dtype)  # [D]
            U_p = layer_bank.U_prior.to(device=device, dtype=dtype)        # [D,K]
            U_e = layer_bank.U_evid.to(device=device, dtype=dtype)         # [D,L]

            x = h_last - mean_p.unsqueeze(0)
            ap = x @ U_p
            proj_p = ap @ U_p.T
            r = x - proj_p
            ae = r @ U_e
            proj_e = ae @ U_e.T

            Ep = (ap ** 2).sum(dim=-1)
            Ee = (ae ** 2).sum(dim=-1)
            logratio = torch.log((Ep + 1e-8) / (Ee + 1e-8))

            tau = float(layer_bank.tau) * tau_scale
            s = max(float(layer_bank.s), 1e-6)
            gate = torch.sigmoid((logratio - tau) / s)  # [B]

            delta = (-beta1 * gate).unsqueeze(-1) * proj_p + (beta2 * gate).unsqueeze(-1) * proj_e

            # norm clamp: avoid repetition/collapse
            r_max = 0.30
            x_norm = torch.norm(h_last, dim=-1, keepdim=True) + 1e-6
            d_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-6
            scale = torch.clamp((r_max * x_norm) / d_norm, max=1.0)
            delta = delta * scale

            new_hidden = hidden
            # avoid clone unless needed (perf): write on a copy only if grad, but no_grad here
            new_hidden = new_hidden.clone()
            new_hidden[:, -1, :] = new_hidden[:, -1, :] + delta

            if debug_once and (not did_debug["v"]):
                print(
                    f"[ORBIT-debug] decoder_idx={decoder_idx} fam={fam} "
                    f"logratio={float(logratio.mean()):.3f} gate={float(gate.mean()):.3f} "
                    f"E_p={float(Ep.mean()):.2f} E_e={float(Ee.mean()):.2f} tau={tau:.3f} s={s:.3f}"
                )
                did_debug["v"] = True

            if rest is None:
                return new_hidden
            return (new_hidden,) + rest

        return hook

    def enable_orbit(
        self,
        family: str,
        layer_indices: List[int],
        beta1: float,
        beta2: float,
        tau_scale: float = 1.0,
        apply_decode: bool = True,
        debug_once: bool = False,
    ):
        if self.bank is None:
            raise RuntimeError("No bank loaded. Provide bank_path or build_bank first.")
        self.close_orbit()
        self._family = family if family in self.bank.families else "open_vqa"
        for di in layer_indices:
            h = self.layers[di].register_forward_hook(
                self._make_hook(di, beta1, beta2, tau_scale, apply_decode, debug_once)
            )
            self._handles.append(h)

    # --------- yes/no scoring (logit-only) ---------
    @torch.no_grad()
    def _forward_logits(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(**inputs, return_dict=True, use_cache=False)
        return out.logits  # [1, T, vocab]

    @torch.no_grad()
    def _next_logits(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self._forward_logits(inputs)
        return logits[:, -1, :]  # [1, vocab]

    @torch.no_grad()
    def _seq_logprob_by_concat(self, inputs: Dict[str, torch.Tensor], cand_ids: List[int]) -> float:
        """
        Compute log P(cand_ids | prompt) using one forward on concatenated ids.
        Works even if cand is multi-token.
        """
        if len(cand_ids) == 0:
            return 0.0

        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)
        base_len = int(input_ids.shape[1])

        cand = torch.tensor([cand_ids], device=input_ids.device, dtype=input_ids.dtype)
        input_ids2 = torch.cat([input_ids, cand], dim=1)

        if attn is not None:
            attn2 = torch.cat([attn, torch.ones((attn.shape[0], len(cand_ids)), device=attn.device, dtype=attn.dtype)], dim=1)
        else:
            attn2 = None

        inputs2 = dict(inputs)
        inputs2["input_ids"] = input_ids2
        if attn2 is not None:
            inputs2["attention_mask"] = attn2

        logits2 = self._forward_logits(inputs2)  # [1, base_len+L, vocab]
        logp = F.log_softmax(logits2, dim=-1)

        lp = 0.0
        # token at position base_len+i is predicted by logits at position base_len+i-1
        for i, tid in enumerate(cand_ids):
            pos = base_len + i - 1
            lp += float(logp[0, pos, tid].item())
        return lp

    @torch.no_grad()
    def _yesno_score_from_inputs(self, inputs: Dict[str, torch.Tensor]) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Returns:
          score = lp_yes - lp_no
          lp_yes, lp_no
          extra info
        """
        tok = self.processor.tokenizer

        yes_ids = tok.encode(" Yes", add_special_tokens=False)
        no_ids = tok.encode(" No", add_special_tokens=False)

        extra: Dict[str, Any] = {
            "yes_ids": yes_ids,
            "no_ids": no_ids,
        }

        # Fast path: both single-token and distinct
        if len(yes_ids) == 1 and len(no_ids) == 1 and yes_ids[0] != no_ids[0]:
            next_logits = self._next_logits(inputs)  # [1, vocab]
            logp = F.log_softmax(next_logits, dim=-1)
            lp_yes = float(logp[0, yes_ids[0]].item())
            lp_no = float(logp[0, no_ids[0]].item())
            score = lp_yes - lp_no
            extra["mode"] = "next_logits"
            return score, lp_yes, lp_no, extra

        # Robust path: concat forward (2 forwards total)
        lp_yes = self._seq_logprob_by_concat(inputs, yes_ids)
        lp_no = self._seq_logprob_by_concat(inputs, no_ids)
        score = lp_yes - lp_no
        extra["mode"] = "concat_forward"
        return score, lp_yes, lp_no, extra

    # --------- generation ---------
    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        question: str,
        enable_orbit: bool,
        layer_indices: List[int],
        beta1: float = 10.0,
        beta2: float = 0.0,
        tau_scale: float = 1.0,
        max_new_tokens: int = 64,
        do_yesno: bool = False,
        use_cache: bool = True,
        apply_decode: bool = True,
        debug_once: bool = False,
        return_info: bool = False,
        # NEW: logit-only / veto
        logit_only: bool = False,
        use_veto: bool = False,
        veto_delta: float = 0.0,
        null_size: int = 336,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        fam = infer_family(question)

        orig_question = question
        if do_yesno and "answer yes or no" not in (question or "").lower():
            question = question.rstrip("?") + "? Answer Yes or No."

        # enable / disable ORBIT
        if enable_orbit:
            self.enable_orbit(
                family=fam,
                layer_indices=layer_indices,
                beta1=beta1,
                beta2=beta2,
                tau_scale=tau_scale,
                apply_decode=apply_decode,
                debug_once=debug_once,
            )
        else:
            self.close_orbit()

        prompt = llava_prompt(question)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if k == "pixel_values":
                    inputs[k] = v.to(device=self.input_device, dtype=self.model.dtype)
                else:
                    inputs[k] = v.to(device=self.input_device)

        info: Dict[str, Any] = {
            "family": fam,
            "enable_orbit": bool(enable_orbit),
            "layer_indices": list(layer_indices) if layer_indices is not None else None,
            "beta1": float(beta1),
            "beta2": float(beta2),
            "tau_scale": float(tau_scale),
            "use_cache": bool(use_cache),
            "apply_decode": bool(apply_decode),
            "do_yesno": bool(do_yesno),
            "logit_only": bool(logit_only),
            "use_veto": bool(use_veto),
            "veto_delta": float(veto_delta),
            "question": question,
            "orig_question": orig_question,
        }

        # YES/NO branch
        if do_yesno:
            score_r, lp_yes_r, lp_no_r, extra_r = self._yesno_score_from_inputs(inputs)
            pred = "Yes" if score_r > 0 else "No"

            info.update({
                "yesno_score_real": float(score_r),
                "lp_yes_real": float(lp_yes_r),
                "lp_no_real": float(lp_no_r),
                "yesno_extra_real": extra_r,
            })

            vetoed = False
            score_n = None
            gap = None

            if use_veto and pred == "Yes":
                null_img = make_null_image(size=null_size)
                inputs_n = self.processor(text=prompt, images=null_img, return_tensors="pt")
                for k, v in inputs_n.items():
                    if torch.is_tensor(v):
                        if k == "pixel_values":
                            inputs_n[k] = v.to(device=self.input_device, dtype=self.model.dtype)
                        else:
                            inputs_n[k] = v.to(device=self.input_device)

                score_n, lp_yes_n, lp_no_n, extra_n = self._yesno_score_from_inputs(inputs_n)
                gap = score_r - score_n
                if gap < veto_delta:
                    pred = "No"
                    vetoed = True

                info.update({
                    "yesno_score_null": float(score_n),
                    "lp_yes_null": float(lp_yes_n),
                    "lp_no_null": float(lp_no_n),
                    "yesno_extra_null": extra_n,
                    "veto_gap": float(gap),
                    "vetoed": bool(vetoed),
                })

            # logit_only: return pred only (no generate)
            if logit_only or max_new_tokens <= 0:
                self.close_orbit()
                return (pred, info) if return_info else pred

            # otherwise (optional) generate some text
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )
            decoded = self.processor.tokenizer.decode(out[0], skip_special_tokens=True)
            ans = strip_assistant(decoded)

            self.close_orbit()
            return (ans, info) if return_info else ans

        # Free-form generation
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )
        decoded = self.processor.tokenizer.decode(out[0], skip_special_tokens=True)
        ans = strip_assistant(decoded)

        self.close_orbit()
        return (ans, info) if return_info else ans

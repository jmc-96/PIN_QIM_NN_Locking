# %% 0) Uvoz i učitavanje modela
import os, sys, torch, numpy as np
from copy import deepcopy

import B_CNN_ucitavanje as UNN
sd, ime, model = UNN.ucitavanje_modela()   # <-- tvoj loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Ako koristiš DataParallel/DDP, ovo može postojati:
model_candidates = [("model", model)]
if hasattr(model, "module"):
    model_candidates.append(("model.module", model.module))

# Ako ima EMA varijanti u globalnom scope-u, pokupi ih:
for name, obj in list(globals().items()):
    if ("ema" in name.lower()) and hasattr(obj, "state_dict") and callable(getattr(obj, "eval", None)):
        model_candidates.append((name, obj))

print("Kandidati za evaluaciju:", [n for n,_ in model_candidates])

    
# %% 1) Eval + detekcija glave
import torch.nn as nn

@torch.no_grad()
def eval_top1(model, loader, max_batches=30, device=device):
    model.eval().to(device)
    correct, total, seen = 0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.numel()
        seen += 1
        if seen >= max_batches:
            break
    return 100.0 * correct / max(1, total)

def find_head_module_name(model, loader, device=device):
    """
    Pronađi ime modula koji emituje [N,1000] logits u forwardu.
    Vraća string (putanja u named_modules) i sam modul.
    """
    model.eval().to(device)
    name_map = {m: n for n, m in model.named_modules()}
    hit = {"name": None, "mod": None, "shape": None}

    hooks = []
    def hook_fn(mod, inp, out):
        try:
            shp = tuple(out.shape)
        except Exception:
            return
        # tražimo 2D [N,1000] ili 4D [N,1000,1,1]
        if (len(shp) == 2 and shp[1] == 1000) or (len(shp) == 4 and shp[1] == 1000 and shp[2:] == (1,1)):
            hit["name"] = name_map.get(mod, "<unknown>")
            hit["mod"]  = mod
            hit["shape"]= shp

    # kači hook na sve Linear i Conv2d (da bude brzo)
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            hooks.append(m.register_forward_hook(hook_fn))

    # prođi jedan batch
    images, _ = next(iter(loader))
    _ = model(images.to(device))

    for h in hooks:
        h.remove()

    if hit["mod"] is None:
        raise RuntimeError("Nisam našao modul koji daje [N,1000] u forwardu. (Proveri loader/batch.)")
    return hit["name"], hit["mod"], hit["shape"]

# %% 2) Prođi sve kandidate, izmeri baseline i nađi glavu koju forward koristi
#    (ovo će ti pokazati KOJU instancu eval skripta zapravo koristi po broju koji dobiješ)

# >>>>>>>>>>>>>>>>>>>> EDIT OVDE AKO TI LOADER NIJE 'val_loader' <<<<<<<<<<<<<<<<<<<<<<
val_loader = val_loader  # zameni ako se drugačije zove
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

baselines = {}
heads = {}

for tag, mdl in model_candidates:
    try:
        acc = eval_top1(mdl, val_loader, max_batches=20)
    except Exception as e:
        print(f"[{tag}] eval error:", e)
        continue
    baselines[tag] = acc
    try:
        head_name, head_mod, head_shape = find_head_module_name(mdl, val_loader)
        heads[tag] = (head_name, head_mod, head_shape)
        print(f"[{tag}] baseline@20b = {acc:.2f}% | head: {head_name}  shape={head_shape}")
    except Exception as e:
        print(f"[{tag}] head-find error:", e)

print("\nRezimei:")
for tag in baselines:
    print(f"  {tag:15s}  acc={baselines[tag]:.2f}%  head={heads.get(tag, ('?',None,None))[0]}")

# %% 3) Zero test na kandidat-instanci koja je ≈68.83%
#    Izaberi "tag" koji je dao 68.83%. Ako je samo jedan – uzmi njega.

# >>>>>>>>>>>>>>>>>>>> EDIT: stavi 'tag' na onog kandidata sa 68.83% <<<<<<<<<<<<<<<<<
tag = max(baselines, key=lambda k: abs(baselines[k]-68.83))  # heuristika ako ti je lakše
print("Koristim kandidat:", tag, " baseline=", baselines[tag])
mdl = dict(model_candidates)[tag]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Sačuvaj originalne težine glave, pa postavi na nule IN-PLACE
import torch.nn as nn
mdl.eval().to(device)
head_name, head_mod, head_shape = heads[tag]

with torch.no_grad():
    if isinstance(head_mod, nn.Linear):
        backup = head_mod.weight.detach().clone()
        head_mod.weight.zero_()
        if head_mod.bias is not None:
            backup_b = head_mod.bias.detach().clone()
            head_mod.bias.zero_()
        else:
            backup_b = None
    elif isinstance(head_mod, nn.Conv2d):
        backup = head_mod.weight.detach().clone()
        head_mod.weight.zero_()
        if head_mod.bias is not None:
            backup_b = head_mod.bias.detach().clone()
            head_mod.bias.zero_()
        else:
            backup_b = None
    else:
        raise RuntimeError(f"Neočekivan tip glave: {type(head_mod)}")

# izmeri acc na nekoliko batch-eva
acc_zero = eval_top1(mdl, val_loader, max_batches=10)
print(f"[{tag}] posle ZERO glave: acc@10b = {acc_zero:.2f}%  (treba ~0.1%)")

# vrati original, da ne zagadiš sesiju
with torch.no_grad():
    head_mod.weight.copy_(backup)
    if backup_b is not None and hasattr(head_mod, "bias") and head_mod.bias is not None:
        head_mod.bias.copy_(backup_b)

# %% 4) Ako ZERO test nije pao, automatski probaj sve kandidate i prijavi pada li skor
drops = []
for tag2, mdl2 in model_candidates:
    try:
        head_name2, head_mod2, _ = heads[tag2]
    except KeyError:
        continue

    mdl2.eval().to(device)
    # backup & zero
    with torch.no_grad():
        backup2 = head_mod2.weight.detach().clone()
        head_mod2.weight.zero_()
        backup_b2 = head_mod2.bias.detach().clone() if hasattr(head_mod2, "bias") and head_mod2.bias is not None else None

    try:
        acc2 = eval_top1(mdl2, val_loader, max_batches=10)
        drops.append((tag2, baselines.get(tag2, float('nan')), acc2))
        print(f"[{tag2}] ZERO test acc@10b = {acc2:.2f}%  (baseline {baselines.get(tag2, float('nan')):.2f}%)")
    finally:
        with torch.no_grad():
            head_mod2.weight.copy_(backup2)
            if backup_b2 is not None and hasattr(head_mod2, "bias") and head_mod2.bias is not None:
                head_mod2.bias.copy_(backup_b2)

print("\nSažetak ZERO testova:")
for t,b,a in drops:
    print(f"  {t:15s}  before={b:.2f}%  after_zero={a:.2f}%  drop={b-a:.2f}pp")

# %% 5) Snimi .pth sa “locked” glavom na PRAVOJ instanci

# >>>>>>> Ako si već utvrdio koji je 'tag' pravi, koristi ga ovde <<<<<<
mdl = dict(model_candidates)[tag]
head_name, head_mod, head_shape = heads[tag]
print("Snimaću pth iz kandidata:", tag, "  head:", head_name)

# Primer: npr. postavi znakove na suprotno kao "izmenu" (samo demonstracija)
with torch.no_grad():
    head_mod.weight.mul_(-1.0)  # <- OVDE umećeš svoju izmenu (ili vrati locked W)

# napravi novi state_dict (na CPU) i snimi
sd_re = {k: v.detach().cpu().clone() for k,v in mdl.state_dict().items()}
out_pth = os.path.abspath("model_locked_sd.pth")
torch.save(sd_re, out_pth)
print("Snimljeno:", out_pth)

# (opciono) quick-reload provera u svežoj instanci istog arh. (ako UNN ima konstruktor)
# new_sd, new_name, new_model = UNN.ucitavanje_modela()   # ako ovo pravi novu instancu bez re-učitavanja fajla
# new_model.load_state_dict(torch.load(out_pth), strict=True)
# print("Reload OK.")

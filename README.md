Code for https://arxiv.org/abs/2409.08601v2 (ICASSP 2025).

## Environment

```
pip install -r requirement.txt
cd diffusers-0.20.0
pip install -e .
cd ..
```

## Download model
From https://huggingface.co/y-ren16/STAV2A

## Inference
```
bash Inf_sh/inf.sh
```

## Eval
```
# For FD, FAD, IS, and KL
bash EvalTools/Eval_audio.sh
# For AV-Align, AA-Align, PAM, and CLAP
bash EvalTools/Eval_all.sh
```
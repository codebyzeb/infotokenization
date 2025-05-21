# Script to launch training byte-level LLM on personal VM

# Only train 99000 rows for small byte-level LLM
#uv run cli.py data finewebedu-download bytelevel-subset --num-train-rows 99000

model=fw57M-tied

uv run scripts/train.py \
    +callbacks.grad_accum.scheduling="{0: 4}" \
    data.eval_batch_size=128 \
    model=$model \
    pwd=/home/zg258/projects/infotokenization \
    torch_compile=true \
    trainer.devices=1 \
    data.num_workers=12 \
    hydra.run.dir=outputs/$model \
    run_folder=. \
    resume_from_checkpoint=.checkpoints/last.ckpt \
    tok_name=bytelevel
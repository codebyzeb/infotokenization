# Script to launch training on personal VM

uv run cli.py data downloda

model=fw57M-tied

uv run scripts/train.py \
    +callbacks.grad_accum.scheduling="{0: 4}" \
    data.eval_batch_size=128 \
    model=$model \
    pwd=/home/zg258/rds/hpc-work/infotokenization \
    torch_compile=true \
    trainer.devices=1 \
    data.num_workers=12 \
    hydra.run.dir=outputs/$model \
    run_folder=. \
    resume_from_checkpoint=.checkpoints/last.ckpt \
    tok_name=bytelevel
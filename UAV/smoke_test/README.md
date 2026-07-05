# DPO Smoke Test

这个文件夹用于在正式 DPO 前做 RTX 5090 32GB 烟测。

烟测目标：

- 确认 Stage I SFT checkpoint 能被 DPO 正确加载。
- 确认 policy model + reference model 在 5090 32GB 上不会 OOM。
- 确认 DPO backward、LoRA 梯度、保存流程都能跑通。
- 避免直接用正式 DPO 跑几个小时后才暴露配置或显存问题。

## 启动方式

在 AutoDL 服务器的项目根目录执行：

```bash
cd /root/Projects/UAV/UAV
conda activate uavmllm
bash smoke_test/run_dpo_smoke.sh
tail -f dpo_smoke.log
```

默认会使用：

```text
SFT checkpoint : /root/autodl-tmp/outputs/stage1_sft_final
source data    : /root/autodl-tmp/data/full5000
smoke data     : /root/autodl-tmp/data/dpo_smoke
smoke pairs    : 160
config         : smoke_test/dpo_smoke_5090.yaml
log            : dpo_smoke.log
```

160 条 DPO pair、`grad_accum=16`、`epochs=1` 大约对应 10 个 optimizer step。

## 通过标准

看到下面这些情况就可以认为烟测通过：

- 没有 CUDA OOM。
- `loss_dpo` 不是 `nan`。
- `loss_ctl` 没有持续爆炸到几十或上百。
- `grad_norm_lora_total` 有正常数值。
- 能保存最终目录：

```text
/root/autodl-tmp/outputs/dpo_smoke/stage2_dpo_final
```

## 烟测通过后启动正式 DPO

```bash
nohup python src/training/train_dpo.py \
  --config configs/rtx5090.yaml \
  --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final \
  --data_dir /root/autodl-tmp/data/full5000 \
  > dpo.log 2>&1 &

echo "PID: $!"
tail -f dpo.log
```

## 可选参数

如果想多跑一点烟测 pair：

```bash
SMOKE_PAIRS=320 bash smoke_test/run_dpo_smoke.sh
```

如果 SFT checkpoint 或数据目录不同：

```bash
SFT_CKPT=/root/autodl-tmp/outputs/stage1_sft_final \
SOURCE_DATA_DIR=/root/autodl-tmp/data/full5000 \
bash smoke_test/run_dpo_smoke.sh
```

## 注意

不要用 `configs/smoke.yaml` 直接做 5090 DPO 烟测。那个配置更偏 96GB 路线，包含 `use_4bit: false`、`max_seq_length: 3456`、`mu: 0.05`，在 RTX 5090 32GB 上更容易 OOM。

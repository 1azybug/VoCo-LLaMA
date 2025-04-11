记得把.gitigore补回来

**playground/data/**
**/checkpoints/**
**__pycache__**
**/hf_models/**
**/hf_datas/**


1. 下载仓库


2. Install Package

```Shell
cd VoCo-LLaMA
conda create -n voco python=3.10 -y
conda activate voco
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```


3. Install additional packages for training cases

```
pip install -e ".[train]"
```

4. 找到conda环境里的hf代码:`miniconda3/envs/voco/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py`
把`VoCo-LLaMA/llava/model/language_model/cache_py/modeling_attn_mask_utils.py`文件复制过去(直接覆盖)
```
cp VoCo-LLaMA/llava/model/language_model/cache_py/modeling_attn_mask_utils.py /data/miniconda3/envs/voco/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py
```

5. 重新安装deepspeed
```
pip install deepspeed==0.15.4
```

6. 训练
```
bash scripts/finetune_voco_llama.sh
```

7. 评估
```
pip install openpyxl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/eval/mmbench.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/eval/sqa.sh
```

8. 提交结果(只有sqa可以直接出结果,其他两个应该是闭源评测)
```
把VoCo-LLaMA/playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015/voco_llava.json提交到https://eval.ai/web/challenges/challenge-page/830/my-submission


把VoCo-LLaMA/playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/voco_llama.xlsx提交到https://mmbench.opencompass.org.cn/mmbench-submission
```

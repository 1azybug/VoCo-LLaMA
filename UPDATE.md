# 复现遇到的问题
1. peft版本太高
```
pip install peft==0.6.0
```

2. zero3.json必须有`"train_batch_size"`字段

3. cuda版本和deepspeed不对应
```
找对应的torch库和deepspeed库
```

4. deepseek给的zero3.json文件用了cpu的优化器
```
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none",
            "pin_memory": true
        },

```

5. no sync context manager is incompatible with gradientpartitioning logic of ZeRo stage 3
```
# 某些时候百度比AI好用
pip install deepspeed==0.15.4
```

6. zero3.json
```

{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "gradient_accumulation_steps": 16,
    "train_micro_batch_size_per_gpu": 1,
    "train_batch_size": 128, 
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "wall_clock_breakdown": false
}

```

7. 下载全部ocr_vqa图片的方法
```
https://github.com/haotian-liu/LLaVA/issues/1618
```

8. 保存模型时报错，需要在lmsys/vicuna-7b-v1.5里的generation_config.json里
因为评估时是贪婪搜索，所以把下面的两行删掉
```
  "temperature": 0.9,
  "top_p": 0.6,
```

# 评估复现的坑

1. checkpoint的文件名要包含llava
2. LlamaModel的forward函数没有处理输入Token只有一个的情况(推理时,第二次前向,输入Token只有一个),为了兼容输入token只有一个都情况下做出如下修改
```
# 不过很奇怪的是,他居然考虑到voco_loc_back要+1

https://github.com/Yxxxb/VoCo-LLaMA/blob/385e7974a866cf73f1cabc8c29cb7a2180fd4dfd/llava/model/language_model/llava_llama_1stg.py#L271

改成

# 整体操作是我每次前向都创建整个序列的mask，管你有没有KVCache
attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask,
    (batch_size, seq_length + past_key_values_length),  # 原来是(batch_size, seq_length), 现在我能保证走同一条路了
    inputs_embeds,  # 这个只用.dtype和isinstance，所以传这个没有影响
    0, # 原来是past_key_values_length
)
# ------------------------------------------
# https://github.com/Yxxxb/VoCo-LLaMA/blob/385e7974a866cf73f1cabc8c29cb7a2180fd4dfd/llava/model/language_model/llava_llama_1stg.py#L305

上面加入
        
# 处理完Attention_mask后    
attention_mask = attention_mask[:,:,-seq_length:,:]
```



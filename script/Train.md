
  - 训练目标改成更偏 precision
    - script/losses.py:125 起，CrackLoss/CrackLossWithAux 支持配置
    - 默认把 Tversky 改成 alpha=0.6, beta=0.4，更重罚 FP
    - 同时开放：
        - --bce-weight
      - --dice-weight
      - --tversky-weight
      - --boundary-weight
      - --detail-weight
      - --tversky-alpha
      - --tversky-beta
      - --tversky-gamma
  - 训练时自动搜阈值
    - script/train.py:212 起新增训练阶段的 search_best_threshold()
    - 每轮验证后自动扫阈值区间
    - 排序规则是：
        - 如果 --selection-metric precision：先最大化 precision，再用 IoU 做次排序
      - 如果 --selection-metric iou：先最大化 IoU，再用 precision 做次排序
  - best checkpoint 改成真正按目标指标选
    - script/train.py:404 附近
    - 新增统一的 best.pt
    - best.pt 由 --selection-metric 决定
    - patience 也只由这个主指标驱动
    - best_iou.pt / best_precision.pt / best_dice.pt 仍然保留
  - checkpoint 里记录最佳阈值
    - val_metrics["threshold"] 会被存进 checkpoint
  - 推理默认读取 checkpoint 内阈值
    - script/test.py:50 把 --threshold 默认改成 -1
    - script/test.py:402 附近会在未显式传阈值时，自动用 checkpoint 里的最佳 threshold
    - 你手工传 --threshold 0.6 时仍会覆盖它

  已通过语法检查。

  你现在可以这样训练：                   
                                
  python /home/fs-ai/CrackNet/script/train.py \
    --crack500-only \                    
    --selection-metric precision \          
    --auto-selection-threshold \                         
    --selection-threshold-min 0.3 \         
    --selection-threshold-max 0.9 \                                          
    --selection-threshold-step 0.02 \
    --tversky-alpha 0.6 \          
    --tversky-beta 0.4 \       
    --tversky-weight 0.35 \                                              
    --boundary-weight 0.3 \
    -detail-weight 0.15 

```bash
python /home/fs-ai/CrackNet/script/train.py --crack500-only --selection-metric precision --auto-selection-threshold --selection-threshold-min 0.3 --selection-threshold-max 0.9 --selection-threshold-step 0.02 --tversky-alpha 0.6  --tversky-beta 0.4 --tversky-weight 0.35 --boundary-weight 0.3 --detail-weight 0.15
```
  如果你想更激进地压误检，可以再提高一点：                                                                                             
                                                                                                                                       
  python /home/fs-ai/CrackNet/script/train.py \  
    --crack500-only \        
    --selection-metric precision \                        
    --auto-selection-threshold \
    --tversky-alpha 0.7 \   
    --tversky-beta 0.3 \  
    --tversky-weight 0.3 \  
    --boundary-weight 0.2 \
    --detail-weight 0.1   

  训练完成后，直接这样推理就会自动使用训练时找到的最佳阈值：

```bash
  python /home/fs-ai/CrackNet/script/test.py \
  --weights /home/fs-ai/CrackNet/script/runs/你的run/weights/best.pt \ 
  --image /home/fs-ai/Pictures/你的图片.png
```
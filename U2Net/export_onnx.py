import os
from ultralytics import YOLO
from yolo_train_single_cls import yolo_name

if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__)) # /home/fs-ai/YOLO-crack-detection/scripts
    
    # 与训练脚本中的 name 参数保持一致
    train_run_name = f"crack_{yolo_name}_single_cls_optimized"

    # 自动拼接出训练生成的最佳模型路径
    model_path = os.path.join(current_dir, "runs/train", train_run_name, "weights/best.pt")

    print(f"正在加载模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        # 列出已有的训练运行，帮助用户定位
        runs_dir = os.path.join(current_dir, "runs/train")
        if os.path.exists(runs_dir):
            available = [d for d in os.listdir(runs_dir)
                         if os.path.exists(os.path.join(runs_dir, d, "weights/best.pt"))]
            if available:
                print(f"可用的训练运行: {available}")
            else:
                print("未找到任何包含 best.pt 的训练运行。")
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    model = YOLO(model_path)
    
    print("开始导出 ONNX 模型，请稍候...")
    
    # 导出参数需与训练时的 imgsz 保持一致
    # 注意：虽然训练时用了 retina_masks=True，但导出为 ONNX 时该参数通常不生效，
    # 因为 ONNX 导出的是模型结构，掩码的后处理（retina_masks）通常在推理阶段进行。
    # 不过，为了确保输入尺寸正确，必须设置 imgsz=416
    success = model.export(
        format="onnx",
        imgsz=416,         # 🔥 必须与 yolo-train.py 中的 imgsz 保持一致
        opset=11,          # 兼容性较好的 opset 版本
        simplify=True,     # 简化模型结构
        dynamic=False,     # 固定输入尺寸（推荐用于嵌入式设备）
        half=False         # 使用 FP32（如果设备支持 FP16，可以改为 True）
    )
    
    print(f"🎉 导出成功! ONNX 文件保存在: {success}")

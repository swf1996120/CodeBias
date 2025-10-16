import os
import shutil

BASE_DIR = "results/SAMPLING"

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file == "result.json":
            old_path = os.path.join(root, file)

            # 上级目录名作为 attribute
            attribute_dir = os.path.basename(root)
            parent_dir = os.path.dirname(root)

            # 正确的新路径：把 attribute 目录去掉，改成 attribute_result.json
            new_path = os.path.join(parent_dir, f"{attribute_dir}_result.json")

            # 如果新路径已存在，避免覆盖
            if os.path.exists(new_path):
                print(f"⚠️ 已存在，跳过: {new_path}")
                continue

            # 创建父目录并移动文件
            try:
                shutil.move(old_path, new_path)
                # 如果 attribute 文件夹为空，删除
                if not os.listdir(root):
                    os.rmdir(root)
                print(f"✅ 修复: {old_path} → {new_path}")
            except Exception as e:
                print(f"❌ 出错: {old_path} → {e}")
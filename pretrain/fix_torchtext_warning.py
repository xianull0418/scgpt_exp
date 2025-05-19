#!/usr/bin/env python3
# 处理TorchText警告的补丁脚本

import sys
import os

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

# 设置环境变量以禁用监控屏障
os.environ['TORCH_DISTRIBUTED_BARRIER_DISABLE_MONITORING'] = '1'

# 在导入任何可能使用torchtext的模块前禁用警告
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# 执行原始脚本
if __name__ == "__main__":
    # 运行原始模块
    pretrain_script = os.path.join(script_dir, "pretrain_init.py")
    with open(pretrain_script) as f:
        script_content = f.read()
    
    # 设置__file__变量以确保相对导入正常工作
    __file__ = pretrain_script
    exec(script_content) 
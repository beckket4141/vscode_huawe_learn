#!/usr/bin/env python3
"""
Conda环境创建脚本
用于创建指定Python版本和包的conda环境
"""

import subprocess
import sys


def run_command(command):
    """执行命令并打印输出"""
    print(f"执行命令: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True)
        print("命令执行成功!")
        if result.stdout:
            print(f"输出:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        if e.stderr:
            print(f"错误信息:\n{e.stderr}")
        return False


def create_conda_environment(env_name, python_version, packages=None):
    """
    创建conda环境
    
    Args:
        env_name (str): 环境名称
        python_version (str): Python版本
        packages (list): 要安装的包列表
    """
    print(f"开始创建conda环境: {env_name}")
    print(f"Python版本: {python_version}")
    
    # 构建创建环境的命令
    command = f"conda create -n {env_name} python={python_version} -y"
    
    # 添加额外的包
    if packages:
        print(f"将安装的额外包: {', '.join(packages)}")
        command += " " + " ".join(packages)
    
    # 执行命令
    if run_command(command):
        print(f"环境 {env_name} 创建成功!")
        return True
    else:
        print(f"环境 {env_name} 创建失败!")
        return False


def main():
    # 环境配置信息
    ENV_NAME = "myenv"  # 环境名称
    PYTHON_VERSION = "3.12"  # Python版本
    
    # 在这里添加你需要的其他包
    # 格式: ["包名1", "包名2=版本号", "包名3"]
    # 例如: ["numpy", "pandas", "matplotlib", "jupyter"]
    PACKAGES = [
        # 数据处理和科学计算相关包
        # "numpy",
        # "pandas",
        # "scipy",
        
        # 数据可视化相关包
        # "matplotlib",
        # "seaborn",
        
        # 机器学习相关包
        # "scikit-learn",
        
        # 深度学习相关包
        # "tensorflow",
        # "torch",
        
        # 开发工具相关包
        # "jupyter",
        # "notebook",
        
        # 其他常用包
        # "requests",
        # "pillow",
    ]
    
    # 创建conda环境
    create_conda_environment(ENV_NAME, PYTHON_VERSION, PACKAGES)
    
    # 激活环境的说明
    print("\n" + "="*50)
    print("环境创建完成!")
    print(f"要激活环境，请在终端中运行: conda activate {ENV_NAME}")
    print("要退出环境，请在终端中运行: conda deactivate")
    print("="*50)


if __name__ == "__main__":
    main()
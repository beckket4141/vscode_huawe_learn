#!/usr/bin/env python3
"""
Conda环境创建脚本
用于创建指定Python版本和包的conda环境
"""

import subprocess
import sys
import os


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
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        if e.stderr:
            print(f"错误信息:\n{e.stderr}")
        return False, e.stderr


def check_conda():
    """检查conda是否已安装"""
    print("检查conda是否已安装...")
    success, output = run_command("conda --version")
    if success:
        print(f"Conda版本: {output.strip()}")
        return True
    else:
        print("未找到conda，请先安装Anaconda或Miniconda")
        return False


def list_environments():
    """列出当前所有conda环境"""
    print("\n当前conda环境列表:")
    success, output = run_command("conda env list")
    if success:
        print(output)
    else:
        print("无法获取环境列表")


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
    
    # 添加额外的包 - 需要使用引号包围每个包名
    if packages:
        print(f"将安装的额外包: {', '.join(packages)}")
        # 将每个包用引号包围，防止特殊字符导致问题
        quoted_packages = [f'"{pkg}"' for pkg in packages]
        command += " " + " ".join(quoted_packages)
    
    # 执行命令
    success, output = run_command(command)
    if success:
        print(f"环境 {env_name} 创建成功!")
        return True
    else:
        print(f"环境 {env_name} 创建失败!")
        return False


def verify_environment(env_name):
    """验证环境是否创建成功"""
    print(f"\n验证环境 {env_name} 是否可用...")
    success, output = run_command(f"conda activate {env_name} && python --version")
    return success


def main():
    # 检查conda是否安装
    if not check_conda():
        return
    
    # 列出当前环境
    list_environments()
    
    # 获取用户输入
    print("\n" + "="*50)
    print("Conda环境创建工具")
    print("="*50)
    
    ENV_NAME = input("请输入环境名称 (默认: myenv): ").strip() or "myenv"
    PYTHON_VERSION = input("请输入Python版本 (默认: 3.9): ").strip() or "3.9"
    
    print("\n请选择预设包组合:")
    print("1. 数据科学 (numpy, pandas, matplotlib, jupyter, scikit-learn)")
    print("2. 深度学习 (numpy, pandas, matplotlib, jupyter, tensorflow)")
    print("3. Web开发 (flask, requests, beautifulsoup4)")
    print("4. 自定义")
    print("5. 仅Python基础环境")
    
    choice = input("请选择 (默认: 1): ").strip() or "1"
    
    PACKAGES = []
    if choice == "1":
        PACKAGES = ["numpy", "pandas", "matplotlib", "jupyter", "scikit-learn"]
    elif choice == "2":
        PACKAGES = ["numpy", "pandas", "matplotlib", "jupyter", "tensorflow"]
    elif choice == "3":
        PACKAGES = ["flask", "requests", "beautifulsoup4"]
    elif choice == "4":
        custom_packages = input("请输入需要安装的包，用空格分隔 (例如: numpy pandas flask): ").strip()
        if custom_packages:
            PACKAGES = custom_packages.split()
    # choice == "5" 时保持PACKAGES为空
    
    print(f"\n即将创建环境:")
    print(f"  环境名称: {ENV_NAME}")
    print(f"  Python版本: {PYTHON_VERSION}")
    if PACKAGES:
        print(f"  预装包: {', '.join(PACKAGES)}")
    else:
        print(f"  预装包: 无")
    
    confirm = input("\n确认创建? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消创建环境")
        return
    
    # 创建conda环境
    if create_conda_environment(ENV_NAME, PYTHON_VERSION, PACKAGES):
        print("\n" + "="*50)
        print("环境创建完成!")
        print(f"要激活环境，请在终端中运行: conda activate {ENV_NAME}")
        print("要退出环境，请在终端中运行: conda deactivate")
        print("="*50)
    else:
        print("环境创建失败，请检查错误信息")


if __name__ == "__main__":
    main()
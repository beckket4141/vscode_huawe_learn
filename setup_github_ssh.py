#!/usr/bin/env python3
"""
GitHub SSH配置脚本
用于自动配置SSH密钥并连接到GitHub，解决网络连接问题
"""

import os
import sys
import subprocess
import platform


def check_ssh_install():
    """检查SSH是否已安装"""
    try:
        result = subprocess.run(['ssh', '-V'], capture_output=True, text=True, shell=True)
        print(f"✓ SSH已安装: {result.stderr.strip()}")
        return True
    except FileNotFoundError:
        print("✗ 未找到SSH客户端")
        return False


def generate_ssh_key(email):
    """生成SSH密钥对"""
    print("正在生成SSH密钥对...")
    try:
        # 根据操作系统选择适当的密钥类型
        if platform.system() == "Windows":
            # Windows上优先使用ed25519算法
            subprocess.run([
                'ssh-keygen', '-t', 'ed25519', '-C', email, 
                '-f', os.path.expanduser('~/.ssh/id_ed25519'), '-N', '""'
            ], check=True, shell=True)
        else:
            # 其他系统也使用ed25519算法
            subprocess.run([
                'ssh-keygen', '-t', 'ed25519', '-C', email,
                '-f', os.path.expanduser('~/.ssh/id_ed25519'), '-N', ''
            ], check=True)
        
        print("✓ SSH密钥对生成成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ SSH密钥对生成失败")
        return False


def start_ssh_agent():
    """启动ssh-agent"""
    print("正在启动ssh-agent...")
    try:
        if platform.system() == "Windows":
            subprocess.run(['eval', '$(ssh-agent -s)'], shell=True, check=True)
        else:
            subprocess.run(['eval', '$(ssh-agent -s)'], shell=True, check=True)
        print("✓ ssh-agent启动成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ ssh-agent启动失败")
        return False


def add_ssh_key():
    """将SSH私钥添加到ssh-agent"""
    print("正在将SSH私钥添加到ssh-agent...")
    try:
        key_path = os.path.expanduser('~/.ssh/id_ed25519')
        if not os.path.exists(key_path):
            key_path = os.path.expanduser('~/.ssh/id_rsa')
            
        subprocess.run(['ssh-add', key_path], check=True, shell=True)
        print("✓ SSH私钥添加成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ SSH私钥添加失败")
        return False


def display_public_key():
    """显示公钥内容"""
    print("\n请将以下SSH公钥添加到您的GitHub账户:")
    print("=" * 50)
    
    # 尝试读取ed25519公钥
    key_path = os.path.expanduser('~/.ssh/id_ed25519.pub')
    if not os.path.exists(key_path):
        # 如果ed25519不存在，尝试RSA
        key_path = os.path.expanduser('~/.ssh/id_rsa.pub')
    
    try:
        with open(key_path, 'r') as f:
            public_key = f.read().strip()
            print(public_key)
            return public_key
    except FileNotFoundError:
        print("✗ 未找到公钥文件")
        return None


def test_github_connection():
    """测试GitHub SSH连接"""
    print("\n正在测试GitHub SSH连接...")
    try:
        result = subprocess.run([
            'ssh', '-T', 'git@github.com'
        ], capture_output=True, text=True, check=True, shell=True)
        print("✓ GitHub SSH连接测试成功")
        print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        if "Permission denied" in e.stderr:
            print("⚠ SSH密钥未正确添加到GitHub账户")
        else:
            print(f"✗ GitHub SSH连接测试失败: {e.stderr}")
        return False


def configure_ssh_github_port():
    """配置SSH使用HTTPS端口(443)连接GitHub"""
    config_path = os.path.expanduser('~/.ssh/config')
    
    # 创建.ssh目录（如果不存在）
    ssh_dir = os.path.dirname(config_path)
    if not os.path.exists(ssh_dir):
        os.makedirs(ssh_dir)
    
    # 添加GitHub配置到SSH配置文件
    github_config = """
# GitHub配置 - 使用HTTPS端口
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
    PreferredAuthentications publickey
"""
    
    try:
        with open(config_path, 'a') as f:
            f.write(github_config)
        print("✓ 已配置SSH通过443端口连接GitHub")
        return True
    except Exception as e:
        print(f"✗ 配置SSH失败: {e}")
        return False


def main():
    print("GitHub SSH配置脚本")
    print("=" * 30)
    
    # 获取用户邮箱
    email = input("请输入您的GitHub邮箱地址: ").strip()
    if not email:
        print("邮箱地址不能为空")
        return
    
    # 检查SSH安装
    if not check_ssh_install():
        print("请先安装SSH客户端")
        return
    
    # 生成SSH密钥对
    if not generate_ssh_key(email):
        return
    
    # 启动ssh-agent
    if not start_ssh_agent():
        return
    
    # 添加SSH密钥
    if not add_ssh_key():
        return
    
    # 显示公钥
    public_key = display_public_key()
    if not public_key:
        return
    
    print("\n" + "=" * 50)
    print("下一步操作:")
    print("1. 复制上面的SSH公钥")
    print("2. 登录GitHub账户")
    print("3. 进入Settings > SSH and GPG keys")
    print("4. 点击New SSH key")
    print("5. 粘贴公钥并保存")
    print("6. 运行此脚本的测试功能验证连接")
    
    # 询问是否测试连接
    test_choice = input("\n是否现在测试GitHub连接? (y/n): ").strip().lower()
    if test_choice == 'y':
        # 如果连接失败，配置使用443端口
        if not test_github_connection():
            print("\n尝试配置SSH通过443端口连接GitHub...")
            if configure_ssh_github_port():
                print("请再次测试连接或稍后手动测试")


if __name__ == "__main__":
    main()
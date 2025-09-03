# Python 教程第1-2章：基础环境与核心语法

> **本文档涵盖**：第1章 Python开发环境与项目架构、第2章 Python核心语法与编程范式

---

## 第1章：Python 开发环境与项目架构

### 1.1 开发环境搭建

#### 1.1.1 Python 解释器安装
```bash
# 推荐使用 Python 3.9+ 版本
python --version  # 检查版本
pip --version     # 检查包管理器

# Windows安装建议
# 1. 从 python.org 下载官方安装包
# 2. 勾选 "Add Python to PATH"
# 3. 勾选 "Install for all users"
```

#### 1.1.2 虚拟环境管理
```bash
# 创建虚拟环境
python -m venv oam_project_env

# 激活虚拟环境
# Windows:
oam_project_env\Scripts\activate
# Linux/Mac:
source oam_project_env/bin/activate

# 安装项目依赖
pip install numpy pillow imageio scipy openpyxl

# 验证安装
python -c "import numpy, scipy, PIL; print('所有依赖包安装成功')"

# 导出依赖清单
pip freeze > requirements.txt

# 从依赖清单安装
pip install -r requirements.txt
```

#### 1.1.3 IDE 配置与代码风格
```json
// .vscode/settings.json 推荐配置
{
    "python.defaultInterpreterPath": "./oam_project_env/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.rulers": [88],
    "files.insertFinalNewline": true,
    "editor.formatOnSave": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true
}
```

```python
# 代码风格示例
"""
模块文档字符串：这是一个示例模块
展示Python代码风格和结构
"""

import math  # 标准库导入
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple  # 类型注解导入

import numpy as np  # 第三方库导入
import scipy.ndimage
from PIL import Image

from .local_module import LocalClass  # 本地模块导入

# 模块级常量
MAX_IMAGE_SIZE = 4096
DEFAULT_PIXEL_SIZE = 1.25e-5

class ExampleClass:
    """类文档字符串：示例类"""
    
    def __init__(self, name: str, size: int = 100):
        """
        初始化方法
        
        Args:
            name: 对象名称
            size: 尺寸大小，默认100
        """
        self.name = name
        self.size = size
        self._private_var = 0  # 私有变量用下划线开头
    
    def public_method(self, data: List[float]) -> float:
        """
        公共方法示例
        
        Args:
            data: 输入数据列表
            
        Returns:
            处理后的结果
            
        Raises:
            ValueError: 当输入数据为空时
        """
        if not data:
            raise ValueError("输入数据不能为空")
        
        return sum(data) / len(data)
    
    def _private_method(self) -> None:
        """私有方法用下划线开头"""
        pass

def module_function(x: float, y: float) -> float:
    """
    模块级函数示例
    
    Args:
        x, y: 输入参数
        
    Returns:
        计算结果
    """
    return math.sqrt(x**2 + y**2)

if __name__ == "__main__":
    # 脚本入口点
    example = ExampleClass("test", 200)
    result = example.public_method([1.0, 2.0, 3.0])
    print(f"结果: {result}")
```

### 1.2 项目架构理解

#### 1.2.1 分层架构模式详解
```
OAM 项目四层架构设计：

┌─────────────────────────────────────────────────────────────┐
│                  表示层 (Presentation Layer)                │
│                         app.py                             │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  系统参数设置     │   交互式设计      │    批量生产       │   │
│  │ SystemConfig    │ Interactive     │ BatchProduction │   │
│  │     Page        │ DesignPage      │     Page        │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
│  功能：用户界面、事件处理、数据绑定、进度反馈                   │
└─────────────────────────────────────────────────────────────┘
                              │ 调用
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   业务层 (Business Layer)                   │
│                       main_direct.py                       │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │  LGConfig   │ LGGenerator │ Excel处理    │  文件管理    │ │
│  │   配置管理   │   核心生成器  │  批量处理    │   工具函数   │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
│  功能：业务逻辑封装、配置管理、文件操作、批量处理编排              │
└─────────────────────────────────────────────────────────────┘
                              │ 调用
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   算法层 (Algorithm Layer)                   │
│                  generatePhase_G_direct.py                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           LG模式数学计算核心                          │   │
│  │  • 坐标系统建立    • LG模式计算   • 相位处理           │   │
│  │  • 全息图生成      • 数值保护     • 参数验证           │   │
│  └─────────────────────────────────────────────────────┘   │
│  功能：核心数学算法实现、数值计算、参数验证                      │
└─────────────────────────────────────────────────────────────┘
                              │ 调用
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  基础算法层 (Foundation Layer)               │
│              laguerre.py + inverse_sinc.py + liner.py      │
│  ┌─────────────┬─────────────┬─────────────────────────┐   │
│  │  拉盖尔多项式  │   逆sinc函数  │      线性光栅函数        │   │
│  │  laguerre()  │inverse_sinc()│      liner()           │   │
│  └─────────────┴─────────────┴─────────────────────────┘   │
│  功能：基础数学函数实现、底层数值计算                           │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2.2 模块依赖关系图
```python
"""
依赖关系示例与解析：

app.py (表示层)
├── main_direct.py (业务层)
│   ├── generatePhase_G_direct.py (算法层)
│   │   ├── laguerre.py (基础算法)
│   │   ├── inverse_sinc.py (基础算法)
│   │   └── liner.py (基础算法)
│   ├── pathlib (标准库 - 路径处理)
│   ├── json (标准库 - 配置序列化)
│   └── datetime (标准库 - 日志时间戳)
├── tkinter (GUI框架 - 用户界面)
├── PIL (第三方库 - 图像处理)
├── numpy (第三方库 - 数值计算)
└── threading (标准库 - 多线程)

依赖原则：
1. 上层可以依赖下层，下层不能依赖上层
2. 同层模块之间尽量减少依赖
3. 核心算法层保持独立，便于测试和复用
"""

# 正确的导入示例
# 在 app.py 中
from main_direct import LGConfig, LGGenerator, batch_process_excel

# 在 main_direct.py 中  
from generatePhase_G_direct import generatePhase_G_direct

# 在 generatePhase_G_direct.py 中
from laguerre import laguerre
from inverse_sinc import inverse_sinc
from liner import liner
```

### 1.3 代码组织原则

#### 1.3.1 模块化设计实践
```python
# 好的模块设计示例

# utils/math_functions.py - 数学工具模块
"""数学计算工具函数"""
import numpy as np
from typing import Union, Sequence

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    if abs(denominator) < 1e-15:
        return default
    return numerator / denominator

def normalize_array(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    数组归一化
    
    Args:
        arr: 输入数组
        method: 归一化方法 "minmax" 或 "zscore"
    """
    if method == "minmax":
        min_val, max_val = arr.min(), arr.max()
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        return arr
    elif method == "zscore":
        mean_val, std_val = arr.mean(), arr.std()
        if std_val > 0:
            return (arr - mean_val) / std_val
        return arr
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

# core/phase_generator.py - 核心生成器模块  
"""相位生成核心类"""
from typing import List, Tuple, Optional
import numpy as np
from ..utils.math_functions import normalize_array

class PhaseGenerator:
    """相位生成器 - 高内聚，单一职责"""
    
    def __init__(self, H: int, V: int, pixel_size: float):
        self.H = H
        self.V = V  
        self.pixel_size = pixel_size
        self._coordinate_cache = None
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取坐标网格（带缓存）"""
        if self._coordinate_cache is None:
            x = np.linspace(-self.H/2, self.H/2-1, self.H) * self.pixel_size
            y = np.linspace(-self.V/2, self.V/2-1, self.V) * self.pixel_size
            self._coordinate_cache = np.meshgrid(x, y)
        return self._coordinate_cache
    
    def generate_linear_phase(self, direction: float, period: float) -> np.ndarray:
        """生成线性相位"""
        X, Y = self.get_coordinates()
        return 2 * np.pi * (X * np.cos(direction) + Y * np.sin(direction)) / period

# ui/main_window.py - 界面模块
"""主窗口界面类"""
import tkinter as tk
from tkinter import ttk
from ..core.phase_generator import PhaseGenerator

class MainWindow:
    """主窗口类 - 专注界面职责"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.phase_generator = None
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        self.root.title("OAM 全息图生成器")
        self.root.geometry("1200x800")
        
        # 创建界面组件...
        self.create_menu()
        self.create_toolbar()
        self.create_main_panel()
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开配置", command=self.load_config)
        file_menu.add_command(label="保存配置", command=self.save_config)
```

#### 1.3.2 命名约定与最佳实践
```python
# 命名规范完整示例

# 1. 类名：PascalCase（每个单词首字母大写）
class LGHologramGenerator:
    """拉盖尔-高斯全息图生成器"""
    pass

class ImagePreviewFrame:
    """图像预览框架"""
    pass

# 2. 函数和方法名：snake_case（下划线分隔）
def generate_phase_map(coefficients, l_values, p_values):
    """生成相位图"""
    pass

def build_filename_from_parameters():
    """从参数构建文件名"""
    pass

def validate_input_parameters():
    """验证输入参数"""
    pass

# 3. 变量名：snake_case
pixel_size = 1.25e-5
image_width = 1920
phase_offset = 0.0
enable_grating = True

# 4. 常量：UPPER_CASE（全大写，下划线分隔）
MAX_IMAGE_SIZE = 4096
DEFAULT_WAIST_RADIUS = 0.00254
PI = 3.14159265359

# 5. 私有属性和方法：_开头
class ConfigManager:
    def __init__(self):
        self.public_setting = "value"
        self._private_cache = {}        # 私有属性
        self.__very_private = None      # 强私有属性（名称修饰）
    
    def public_method(self):
        """公共方法"""
        return self._internal_calculation()
    
    def _internal_calculation(self):
        """私有方法，仅供内部使用"""
        pass
    
    def __secret_method(self):
        """强私有方法"""
        pass

# 6. 特殊含义的命名
def process_data(input_data):
    # 临时变量
    temp_result = []
    
    # 循环变量
    for i, item in enumerate(input_data):
        for j in range(len(item)):
            # 使用有意义的名称而不是 a, b, c
            current_value = item[j]
            processed_value = current_value * 2
            temp_result.append(processed_value)
    
    return temp_result

# 7. 布尔变量命名（is_, has_, can_, should_）
is_valid = True
has_data = False
can_process = True
should_normalize = False

# 8. 文件和模块命名
"""
文件命名规范：
- config_manager.py (模块名：snake_case)
- image_processor.py
- main_window.py

包命名规范：
- oam_toolkit/
- image_processing/
- user_interface/
"""
```

---

## 第2章：Python 核心语法与编程范式

### 2.1 变量与数据类型深入

#### 2.1.1 基础数据类型在科学计算中的应用
```python
import numpy as np
from typing import Union, List, Dict, Any

# 数值类型的精确使用
def demonstrate_numeric_types():
    """演示数值类型在科学计算中的应用"""
    
    # 整数类型 - 用于索引、计数、标识
    image_width: int = 1920
    image_height: int = 1152
    mode_index: int = 5
    
    # 浮点类型 - 用于物理量、计算结果
    pixel_size: float = 1.25e-5      # 12.5微米，科学计数法
    waist_radius: float = 0.00254     # 2.54毫米
    phase_offset: float = np.pi / 4   # π/4弧度
    
    # 复数类型 - 用于量子态系数、相位表示
    quantum_coefficient: complex = 1.0 + 0.5j
    phase_factor: complex = np.exp(1j * np.pi / 3)  # e^(iπ/3)
    
    # 展示精度差异
    float32_value = np.float32(1.0/3.0)
    float64_value = np.float64(1.0/3.0)
    
    print(f"float32精度: {float32_value}")
    print(f"float64精度: {float64_value}")
    print(f"精度差异: {abs(float64_value - float32_value)}")
    
    return {
        'integers': (image_width, image_height, mode_index),
        'floats': (pixel_size, waist_radius, phase_offset),
        'complex': (quantum_coefficient, phase_factor),
        'precision': (float32_value, float64_value)
    }

# 字符串处理在文件命名中的应用  
def advanced_string_operations():
    """高级字符串操作"""
    
    # 格式化字符串 - 用于动态文件名生成
    def build_filename(amplitude: float, phase: float, p: int, l: int, waist: float) -> str:
        """构建标准化的文件名"""
        
        # 方法1：f-string（推荐，Python 3.6+）
        filename_f = f"LG_{amplitude:.3f},{phase:.2f}pi({p},{l})_w_{waist*1000:.1f}mm.bmp"
        
        # 方法2：format方法
        filename_format = "LG_{:.3f},{:.2f}pi({},{})_w_{:.1f}mm.bmp".format(
            amplitude, phase, p, l, waist*1000
        )
        
        # 方法3：%格式化（旧式，但在某些情况下仍有用）
        filename_percent = "LG_%.3f,%.2fpi(%d,%d)_w_%.1fmm.bmp" % (
            amplitude, phase, p, l, waist*1000
        )
        
        return filename_f  # 返回最现代的格式
    
    # 字符串验证和清理
    def sanitize_filename(raw_name: str) -> str:
        """清理文件名中的非法字符"""
        import re
        
        # 移除或替换非法字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', raw_name)
        
        # 移除多余的空白
        sanitized = re.sub(r'\s+', '_', sanitized.strip())
        
        # 限制长度
        if len(sanitized) > 200:
            name, ext = sanitized.rsplit('.', 1)
            sanitized = name[:195] + '.' + ext
        
        return sanitized
    
    # 路径处理
    from pathlib import Path
    
    def safe_path_join(*parts: str) -> Path:
        """安全的路径拼接"""
        # 使用pathlib处理跨平台路径
        path = Path()
        for part in parts:
            # 清理每个部分
            clean_part = sanitize_filename(part)
            path = path / clean_part
        return path
    
    # 示例使用
    test_filename = build_filename(1.234, 0.75, 2, -1, 0.00254)
    cleaned = sanitize_filename("Invalid<>Name??.bmp")
    safe_path = safe_path_join("output", "holograms", test_filename)
    
    return {
        'original_filename': test_filename,
        'cleaned_filename': cleaned,
        'safe_path': str(safe_path)
    }
```

#### 2.1.2 容器类型与算法应用
```python
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque, namedtuple
import numpy as np

def container_types_in_practice():
    """容器类型在实际项目中的应用"""
    
    # 列表 - 有序、可变，用于存储序列数据
    def list_operations():
        """列表操作示例"""
        
        # 存储量子模式参数
        l_values: List[int] = [1, -1, 2, -2, 3, -3]
        p_values: List[int] = [0, 0, 1, 1, 2, 2]
        coefficients: List[complex] = [
            1.0 + 0j, 1.0 + 0j, 0.5 + 0j, 
            0.5 + 0j, 0.25 + 0j, 0.25 + 0j
        ]
        
        # 列表推导式 - 生成衍生数据
        amplitudes = [abs(coeff) for coeff in coefficients]
        phases = [np.angle(coeff) for coeff in coefficients]
        
        # 配对操作
        mode_pairs = list(zip(l_values, p_values, coefficients))
        
        # 过滤操作 - 只保留非零系数的模式
        active_modes = [(l, p, c) for l, p, c in mode_pairs if abs(c) > 1e-10]
        
        # 排序操作 - 按照|l|值排序
        sorted_modes = sorted(active_modes, key=lambda x: abs(x[0]))
        
        return {
            'original_lists': (l_values, p_values, coefficients),
            'derived_data': (amplitudes, phases),
            'active_modes': active_modes,
            'sorted_modes': sorted_modes
        }
    
    # 元组 - 有序、不可变，用于返回多个值
    def tuple_operations():
        """元组操作示例"""
        
        # 命名元组 - 增强可读性
        ModeConfig = namedtuple('ModeConfig', ['l', 'p', 'coefficient', 'amplitude', 'phase'])
        
        def create_mode_config(l: int, p: int, coeff: complex) -> ModeConfig:
            """创建模式配置"""
            return ModeConfig(
                l=l,
                p=p, 
                coefficient=coeff,
                amplitude=abs(coeff),
                phase=np.angle(coeff)
            )
        
        # 函数返回多个值
        def generate_coordinates(H: int, V: int, pixel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """生成坐标数组"""
            x = np.linspace(-H/2, H/2-1, H) * pixel_size
            y = np.linspace(-V/2, V/2-1, V) * pixel_size
            X, Y = np.meshgrid(x, y)
            rho = np.sqrt(X**2 + Y**2)
            phi = np.angle(X + 1j*Y)
            return X, Y, rho, phi
        
        # 元组解包
        mode1 = create_mode_config(1, 0, 1+0j)
        mode2 = create_mode_config(-1, 0, 1+0j)
        
        # 访问命名元组字段
        print(f"模式1: l={mode1.l}, p={mode1.p}, 幅度={mode1.amplitude:.3f}")
        
        return mode1, mode2
    
    # 字典 - 键值对，用于配置和映射
    def dict_operations():
        """字典操作示例"""
        
        # 配置参数管理
        config: Dict[str, Any] = {
            'image_size': {'H': 1920, 'V': 1152},
            'optical': {
                'default_waist': 0.00254,
                'enable_correction': True,
                'pixel_size': 1.25e-5
            },
            'grating': {
                'weight': -1.0,
                'period': 12.0,
                'enable': True
            },
            'output': {
                'format': 'BMP',
                'directory': 'LG_output',
                'prefix': 'LG_'
            }
        }
        
        # 安全的字典访问
        def get_config_value(config: Dict, path: str, default=None):
            """通过路径安全获取配置值"""
            keys = path.split('.')
            current = config
            
            try:
                for key in keys:
                    current = current[key]
                return current
            except (KeyError, TypeError):
                return default
        
        # 使用示例
        waist = get_config_value(config, 'optical.default_waist', 0.001)
        format_type = get_config_value(config, 'output.format', 'PNG')
        
        # 字典推导式
        active_settings = {
            key: value for key, value in config['optical'].items() 
            if isinstance(value, bool) and value
        }
        
        # defaultdict - 自动创建缺失的键
        mode_statistics = defaultdict(int)
        modes = [(1, 0), (-1, 0), (2, 1), (1, 0)]  # 重复的(1,0)
        
        for l, p in modes:
            mode_statistics[f"l={l},p={p}"] += 1
        
        return {
            'config': config,
            'waist': waist,
            'format': format_type,
            'active_settings': active_settings,
            'statistics': dict(mode_statistics)
        }
    
    # 集合 - 无序、去重，用于成员测试和集合运算
    def set_operations():
        """集合操作示例"""
        
        # 支持的文件格式
        supported_formats: Set[str] = {'BMP', 'PNG', 'TIFF', 'JPEG'}
        
        # 当前使用的格式
        used_formats: Set[str] = {'BMP', 'PNG'}
        
        # 集合运算
        unused_formats = supported_formats - used_formats
        all_image_formats = supported_formats | {'GIF', 'WEBP'}
        common_formats = supported_formats & {'PNG', 'JPEG', 'GIF'}
        
        # 快速成员测试
        def is_format_supported(format_type: str) -> bool:
            return format_type.upper() in supported_formats
        
        # 去重操作
        duplicate_modes = [1, -1, 2, 1, -1, 3, 2]
        unique_modes = list(set(duplicate_modes))
        
        return {
            'supported': supported_formats,
            'unused': unused_formats,
            'all_formats': all_image_formats,
            'common': common_formats,
            'unique_modes': unique_modes
        }
    
    return {
        'lists': list_operations(),
        'tuples': tuple_operations(),
        'dicts': dict_operations(),
        'sets': set_operations()
    }
```

### 2.2 控制流与逻辑结构

#### 2.2.1 条件语句的科学计算应用
```python
import math
import numpy as np
from typing import Union, Optional

def conditional_statements_in_science():
    """条件语句在科学计算中的应用"""
    
    # 多分支条件：根据模式参数选择计算方法
    def calculate_beam_waist(w_base: float, l: int, p: int, correction_mode: int) -> float:
        """
        根据模式参数计算光腰半径
        
        Args:
            w_base: 基础光腰半径
            l: 角量子数
            p: 径向量子数  
            correction_mode: 修正模式 0=自适应, 1=固定, 2=自定义
        """
        if correction_mode == 0:
            # 自适应光腰：考虑模式阶数
            factor = math.sqrt(abs(l) + 2*p + 1)
            return w_base / factor if factor > 0 else w_base
            
        elif correction_mode == 1:
            # 固定光腰：不做修正
            return w_base
            
        elif correction_mode == 2:
            # 自定义修正：可以添加其他算法
            custom_factor = 1.0 + 0.1 * abs(l) + 0.05 * p
            return w_base / custom_factor
            
        else:
            raise ValueError(f"不支持的光腰修正模式: {correction_mode}")
    
    # 复杂条件判断：参数验证
    def validate_generation_parameters(H: int, V: int, coeffs: list, l_list: list, p_list: list) -> tuple:
        """
        验证全息图生成参数
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 图像尺寸验证
        if H <= 0 or V <= 0:
            errors.append("图像尺寸必须为正整数")
        elif H > 4096 or V > 4096:
            errors.append("图像尺寸过大，可能导致内存不足")
        
        # 参数列表长度验证
        if not (len(coeffs) == len(l_list) == len(p_list)):
            errors.append(f"参数列表长度不一致: coeffs({len(coeffs)}), l_list({len(l_list)}), p_list({len(p_list)})")
        
        # 空参数检查
        if len(coeffs) == 0:
            errors.append("至少需要一个模式参数")
        
        # 物理参数验证
        for i, (coeff, l, p) in enumerate(zip(coeffs, l_list, p_list)):
            if not isinstance(l, int):
                errors.append(f"第{i}个角量子数l必须为整数，得到: {type(l).__name__}")
            
            if not isinstance(p, int) or p < 0:
                errors.append(f"第{i}个径向量子数p必须为非负整数，得到: {p}")
            
            if not isinstance(coeff, (int, float, complex)):
                errors.append(f"第{i}个系数类型无效: {type(coeff).__name__}")
        
        # 数值范围检查
        if len(coeffs) > 0:
            max_amplitude = max(abs(c) for c in coeffs)
            if max_amplitude == 0:
                errors.append("所有系数的幅度都为零")
            elif max_amplitude > 1e10:
                errors.append(f"系数幅度过大: {max_amplitude}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    # 条件表达式（三元运算符）
    def safe_operations():
        """安全操作示例"""
        
        def safe_divide(a: float, b: float) -> float:
            """安全除法"""
            return a / b if abs(b) > 1e-15 else 0.0
        
        def safe_sqrt(x: float) -> float:
            """安全开方"""
            return math.sqrt(x) if x >= 0 else 0.0
        
        def safe_log(x: float) -> float:
            """安全对数"""
            return math.log(x) if x > 0 else float('-inf')
        
        # 条件赋值
        def process_image_data(data: np.ndarray, normalize: bool = True) -> np.ndarray:
            """处理图像数据"""
            
            # 数据清理
            cleaned_data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
            
            # 条件归一化
            if normalize and cleaned_data.max() > cleaned_data.min():
                min_val, max_val = cleaned_data.min(), cleaned_data.max()
                normalized_data = (cleaned_data - min_val) / (max_val - min_val) * 255
            else:
                normalized_data = cleaned_data
            
            # 条件类型转换
            result = normalized_data.astype(np.uint8) if normalized_data.max() <= 255 else normalized_data
            
            return result
        
        return {
            'safe_divide': safe_divide(10, 0.5),
            'safe_divide_zero': safe_divide(10, 0),
            'safe_sqrt': safe_sqrt(4),
            'safe_sqrt_negative': safe_sqrt(-4)
        }
    
    # 测试示例
    test_results = {
        'waist_adaptive': calculate_beam_waist(0.00254, 1, 0, 0),
        'waist_fixed': calculate_beam_waist(0.00254, 1, 0, 1),
        'validation_pass': validate_generation_parameters(1920, 1152, [1+0j], [1], [0]),
        'validation_fail': validate_generation_parameters(0, 0, [], [], []),
        'safe_ops': safe_operations()
    }
    
    return test_results
```

#### 2.2.2 循环结构与向量化权衡
```python
import time
import numpy as np
from typing import List, Callable

def loop_vs_vectorization():
    """循环与向量化的对比学习"""
    
    # 拉盖尔多项式的不同实现方式
    
    # 方法1：传统Python循环（清晰但慢）
    def laguerre_python_loops(p: int, l: int, x: np.ndarray) -> np.ndarray:
        """使用Python循环的拉盖尔多项式计算"""
        result = np.zeros_like(x, dtype=float)
        
        # 逐元素计算（教学清晰）
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if p == 0:
                    result[i, j] = 1.0
                elif p == 1:
                    result[i, j] = 1 + l - x[i, j]
                else:
                    # 递推计算
                    L0, L1 = 1.0, 1 + l - x[i, j]
                    for n in range(1, p):
                        Ln = ((2*n + 1 + l - x[i, j]) * L1 - (n + l) * L0) / (n + 1)
                        L0, L1 = L1, Ln
                    result[i, j] = L1
        
        return result
    
    # 方法2：NumPy向量化（高效）
    def laguerre_vectorized(p: int, l: int, x: np.ndarray) -> np.ndarray:
        """向量化的拉盖尔多项式计算"""
        if p == 0:
            return np.ones_like(x)
        elif p == 1:
            return 1 + l - x
        else:
            L0 = np.ones_like(x)
            L1 = 1 + l - x
            
            for n in range(1, p):
                Ln = ((2*n + 1 + l - x) * L1 - (n + l) * L0) / (n + 1)
                L0, L1 = L1, Ln
            
            return L1
    
    # 方法3：混合方法（特定场景优化）
    def laguerre_hybrid(p: int, l: int, x: np.ndarray) -> np.ndarray:
        """混合方法：对小数组使用循环，大数组使用向量化"""
        
        # 阈值：数组大小
        THRESHOLD = 1000
        
        if x.size < THRESHOLD:
            # 小数组使用简单方法
            return laguerre_python_loops(p, l, x)
        else:
            # 大数组使用向量化
            return laguerre_vectorized(p, l, x)
    
    # 性能基准测试
    def benchmark_methods():
        """性能基准测试"""
        
        # 测试数据
        test_cases = [
            ("小数组", np.random.random((50, 50))),
            ("中等数组", np.random.random((200, 200))),
            ("大数组", np.random.random((500, 500))
        ]
        
        methods = {
            'Python循环': laguerre_python_loops,
            '向量化': laguerre_vectorized,
            '混合方法': laguerre_hybrid
        }
        
        results = {}
        
        for case_name, test_array in test_cases:
            results[case_name] = {}
            print(f"\n测试 {case_name} ({test_array.shape}):")
            
            for method_name, method_func in methods.items():
                # 预热
                method_func(2, 1, test_array[:10, :10])
                
                # 计时
                start_time = time.perf_counter()
                result = method_func(2, 1, test_array)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                results[case_name][method_name] = {
                    'time': execution_time,
                    'result_shape': result.shape,
                    'result_mean': result.mean()
                }
                
                print(f"  {method_name}: {execution_time:.4f}秒")
        
        return results
    
    # 向量化的最佳实践
    def vectorization_best_practices():
        """向量化编程最佳实践"""
        
        # 1. 避免显式循环
        def bad_example(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
            """不好的例子：显式循环"""
            result = np.zeros_like(arr1)
            for i in range(arr1.shape[0]):
                for j in range(arr1.shape[1]):
                    result[i, j] = arr1[i, j] ** 2 + arr2[i, j] ** 2
            return result
        
        def good_example(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
            """好例子：向量化操作"""
            return arr1**2 + arr2**2
        
        # 2. 使用布尔索引
        def conditional_operations(data: np.ndarray) -> np.ndarray:
            """条件操作示例"""
            result = data.copy()
            
            # 不好的方式
            # for i in range(data.shape[0]):
            #     for j in range(data.shape[1]):
            #         if data[i, j] > 0.5:
            #             result[i, j] = data[i, j] * 2
            #         else:
            #             result[i, j] = 0
            
            # 好的方式：布尔索引
            mask = data > 0.5
            result[mask] = data[mask] * 2
            result[~mask] = 0
            
            return result
        
        # 3. 广播的有效使用
        def broadcasting_example():
            """广播示例"""
            # 创建坐标网格
            x = np.linspace(-5, 5, 100).reshape(1, -1)  # (1, 100)
            y = np.linspace(-5, 5, 100).reshape(-1, 1)  # (100, 1)
            
            # 广播计算二维函数
            z = np.exp(-(x**2 + y**2))  # 自动广播为 (100, 100)
            
            return x, y, z
        
        # 4. 内存效率
        def memory_efficient_operations():
            """内存高效的操作"""
            
            # 原地操作
            def normalize_inplace(arr: np.ndarray) -> None:
                """原地归一化，节省内存"""
                arr -= arr.mean()  # 原地减法
                arr /= arr.std()   # 原地除法
            
            # 视图vs副本
            def demonstrate_views():
                """演示视图与副本"""
                original = np.random.random((1000, 1000))
                
                # 视图（共享内存，快速）
                view = original[::2, ::2]      # 每隔一个元素
                transposed = original.T        # 转置
                
                # 副本（独立内存，慢但安全）
                copy = original.copy()
                
                print(f"原数组与view共享内存: {np.shares_memory(original, view)}")
                print(f"原数组与copy共享内存: {np.shares_memory(original, copy)}")
                
                return original, view, copy
            
            return demonstrate_views()
        
        return {
            'conditional_ops': conditional_operations(np.random.random((100, 100))),
            'broadcasting': broadcasting_example(),
            'memory_ops': memory_efficient_operations()
        }
    
    return {
        'benchmark': benchmark_methods(),
        'best_practices': vectorization_best_practices()
    }
```

### 2.3 函数设计与模块化

#### 2.3.1 函数签名设计原则
```python
from typing import Optional, Union, Callable, Sequence, Tuple, List, Dict, Any
import numpy as np
import inspect
from functools import wraps

def function_design_principles():
    """函数设计原则与最佳实践"""
    
    # 1. 清晰的函数签名
    def generate_hologram_sequence(
        coefficients: Sequence[complex],           # 使用Sequence支持list/tuple
        l_values: Sequence[int],
        p_values: Sequence[int],
        waist_start: float,
        waist_end: float,
        waist_step: float,
        image_size: Tuple[int, int] = (1152, 1920),  # 默认参数
        pixel_size: float = 1.25e-5,
        enable_grating: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,  # 回调函数
        output_format: str = "BMP"
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:  # 明确返回类型
        """
        生成光腰扫描的全息图序列
        
        这是一个完整的函数文档字符串示例，展示了所有必要的信息。
        
        Args:
            coefficients: 模式复系数列表，长度应与 l_values, p_values 一致
                例如: [1+0j, 1+0j] 表示两个模式的等权重叠加
            l_values: 角量子数列表，可包含正负整数
                例如: [1, -1] 表示 l=+1 和 l=-1 模式
            p_values: 径向量子数列表，必须为非负整数
                例如: [0, 0] 表示基模
            waist_start: 扫描起始光腰半径 (米)，建议范围 1e-4 到 1e-2
            waist_end: 扫描结束光腰半径 (米)，必须大于 waist_start
            waist_step: 光腰步长 (米)，建议不小于 1e-5
            image_size: 图像尺寸 (V, H)，默认 1152×1920
            pixel_size: 像素物理尺寸 (米)，默认 12.5μm
            enable_grating: 是否启用线性光栅，影响衍射效率
            progress_callback: 进度回调函数，签名为 callback(current, total, message)
                如果提供，将在每个光腰值计算完成时调用
            output_format: 输出格式，支持 "BMP"|"PNG"|"TIFF"
        
        Returns:
            列表，每个元素为 (光腰值, 相位图, 全息图) 的三元组
            - 光腰值: float，单位米
            - 相位图: np.ndarray，shape=(V,H)，dtype=uint8，范围0-255
            - 全息图: np.ndarray，shape=(V,H)，dtype=uint8，范围0-255
        
        Raises:
            ValueError: 参数验证失败时
                - 系数、l值、p值列表长度不一致
                - 光腰参数无效（start >= end, step <= 0）
                - 图像尺寸无效（<= 0 或过大）
            RuntimeError: 计算过程出错时
                - 内存不足
                - 数值计算异常
        
        Example:
            >>> coeffs = [1+0j, 1+0j]
            >>> l_vals = [1, -1]
            >>> p_vals = [0, 0]
            >>> results = generate_hologram_sequence(
            ...     coeffs, l_vals, p_vals, 1e-3, 5e-3, 0.5e-3
            ... )
            >>> print(f"生成了 {len(results)} 组图像")
            生成了 9 组图像
            
        Note:
            - 函数会自动验证输入参数的有效性
            - 大图像尺寸可能需要较长计算时间
            - 建议使用progress_callback监控长时间计算
            
        See Also:
            generate_single_hologram: 生成单个全息图
            save_hologram_sequence: 保存全息图序列到文件
        """
        
        # 参数验证（示例实现）
        if len(coefficients) != len(l_values) or len(coefficients) != len(p_values):
            raise ValueError(f"参数长度不一致: coeffs({len(coefficients)}), l({len(l_values)}), p({len(p_values)})")
        
        if waist_start >= waist_end:
            raise ValueError(f"起始光腰({waist_start}) 必须小于结束光腰({waist_end})")
        
        if waist_step <= 0:
            raise ValueError(f"光腰步长({waist_step}) 必须为正数")
        
        V, H = image_size
        if V <= 0 or H <= 0:
            raise ValueError(f"图像尺寸必须为正数: {image_size}")
        
        # 实际实现会在这里...
        results = []
        waist_values = np.arange(waist_start, waist_end + waist_step/2, waist_step)
        
        for i, waist in enumerate(waist_values):
            if progress_callback:
                progress_callback(i, len(waist_values), f"计算光腰 {waist*1000:.1f}mm")
            
            # 这里会调用实际的计算函数
            phase_map = np.random.randint(0, 256, (V, H), dtype=np.uint8)  # 示例
            hologram = np.random.randint(0, 256, (V, H), dtype=np.uint8)   # 示例
            
            results.append((waist, phase_map, hologram))
        
        return results
    
    # 2. 参数验证装饰器
    def validate_types(**type_hints):
        """类型验证装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 获取函数签名
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # 验证类型
                for param_name, expected_type in type_hints.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if not isinstance(value, expected_type):
                            raise TypeError(
                                f"{func.__name__}.{param_name} 期望 {expected_type.__name__}, "
                                f"得到 {type(value).__name__}"
                            )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # 3. 范围验证装饰器
    def validate_ranges(**range_specs):
        """数值范围验证装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                for param_name, (min_val, max_val) in range_specs.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if not (min_val <= value <= max_val):
                            raise ValueError(
                                f"{func.__name__}.{param_name}={value} 超出范围 [{min_val}, {max_val}]"
                            )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # 4. 使用装饰器的函数示例
    @validate_types(H=int, V=int, pixel_size=float)
    @validate_ranges(H=(1, 4096), V=(1, 4096), pixel_size=(1e-8, 1e-3))
    def create_coordinate_grid(H: int, V: int, pixel_size: float = 1.25e-5) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建坐标网格
        
        Args:
            H: 水平像素数，范围 1-4096
            V: 垂直像素数，范围 1-4096  
            pixel_size: 像素大小，范围 1e-8 到 1e-3 米
            
        Returns:
            (X, Y): 坐标网格数组
        """
        x = np.linspace(-H/2, H/2-1, H) * pixel_size
        y = np.linspace(-V/2, V/2-1, V) * pixel_size
        X, Y = np.meshgrid(x, y)
        return X, Y
    
    # 5. 函数重载（使用typing.overload）
    from typing import overload
    
    @overload
    def process_image(image: np.ndarray) -> np.ndarray: ...
    
    @overload  
    def process_image(image: np.ndarray, normalize: bool) -> np.ndarray: ...
    
    @overload
    def process_image(image: np.ndarray, normalize: bool, output_type: type) -> np.ndarray: ...
    
    def process_image(image: np.ndarray, normalize: bool = True, output_type: type = np.uint8) -> np.ndarray:
        """
        处理图像（重载示例）
        
        支持多种调用方式：
        - process_image(img)
        - process_image(img, normalize=False)  
        - process_image(img, True, np.float32)
        """
        if normalize:
            image = (image - image.min()) / (image.max() - image.min())
        
        if output_type == np.uint8:
            return (image * 255).astype(np.uint8)
        else:
            return image.astype(output_type)
    
    return {
        'coordinate_grid': create_coordinate_grid(100, 100),
        'processed_image': process_image(np.random.random((50, 50)))
    }
```

---

**文档状态**: 第1-2章详细内容已完成 ✅

**包含内容**:
- 开发环境搭建与配置
- 项目架构理解  
- 代码组织原则
- 数据类型详解
- 控制流与循环优化
- 函数设计最佳实践

**下一步**: 准备创建第3-4章（面向对象编程与NumPy数组计算）

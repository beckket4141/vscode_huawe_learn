# Python 教程第3-4章：面向对象编程与NumPy数组计算

> **本文档涵盖**：第3章 面向对象编程与设计模式、第4章 NumPy数组计算与数学建模

---

## 第3章：面向对象编程与设计模式

### 3.1 类设计原则与实践

#### 3.1.1 单一职责原则 (SRP) 在项目中的应用
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json

# 不好的设计：职责混乱的类
class BadLGProcessor:
    """反面教材：一个类承担太多职责"""
    def __init__(self):
        self.config = {}
        self.ui_window = None
        self.image_data = None
    
    def load_config(self):           # 配置管理
        pass
    def generate_hologram(self):     # 算法计算  
        pass
    def save_to_file(self):         # 文件操作
        pass
    def update_progress_bar(self):   # UI更新
        pass
    def validate_input(self):       # 参数验证
        pass

# 好的设计：职责分离
class LGConfig:
    """专门负责配置管理的类"""
    
    def __init__(self):
        # 图像参数
        self.H: int = 1920
        self.V: int = 1152
        self.pixel_size: float = 1.25e-5
        
        # 光学参数
        self.default_waist: float = 0.00254
        self.waist_correction: bool = True
        
        # 光栅参数
        self.enable_grating: bool = True
        self.grating_weight: float = -1.0
        self.grating_period: float = 12.0
        
        # 输出设置
        self.output_dir: str = "LG_output"
        self.filename_prefix: str = "LG_"
        self.save_format: str = "BMP"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化"""
        return {
            'H': self.H, 'V': self.V, 'pixel_size': self.pixel_size,
            'default_waist': self.default_waist,
            'waist_correction': self.waist_correction,
            'enable_grating': self.enable_grating,
            'grating_weight': self.grating_weight,
            'grating_period': self.grating_period,
            'output_dir': self.output_dir,
            'filename_prefix': self.filename_prefix,
            'save_format': self.save_format
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载配置"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def load_from_file(self, filepath: Path) -> None:
        """从文件加载配置"""
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.from_dict(data)
        else:
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
    
    def save_to_file(self, filepath: Path) -> None:
        """保存配置到文件"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

class LGGenerator:
    """专门负责算法计算的类"""
    
    def __init__(self, config: LGConfig):
        self.config = config
        self._coordinate_cache = None
    
    def generate_single(self, coeffs: List[complex], l_list: List[int], 
                       p_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """生成单个全息图和相位图"""
        # 这里会调用底层算法
        from generatePhase_G_direct import generatePhase_G_direct
        
        return generatePhase_G_direct(
            H=self.config.H, V=self.config.V,
            w=self.config.default_waist,
            wd=[], coeffs=coeffs, l_list=l_list, p_list=p_list,
            r=self.config.grating_weight,
            k=self.config.grating_period,
            nn=0 if self.config.enable_grating else 1,
            m=0 if self.config.waist_correction else 1,
            pixel_size=self.config.pixel_size
        )
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取坐标网格（带缓存）"""
        if self._coordinate_cache is None:
            x = np.linspace(-self.config.H/2, self.config.H/2-1, self.config.H) * self.config.pixel_size
            y = np.linspace(-self.config.V/2, self.config.V/2-1, self.config.V) * self.config.pixel_size
            self._coordinate_cache = np.meshgrid(x, y)
        return self._coordinate_cache

class FileManager:
    """专门负责文件操作的类"""
    
    @staticmethod
    def save_image(filepath: Path, image_data: np.ndarray) -> None:
        """保存图像文件"""
        try:
            import imageio.v2 as imageio
            imageio.imwrite(str(filepath), image_data)
        except ImportError:
            from PIL import Image
            Image.fromarray(image_data).save(str(filepath))
    
    @staticmethod
    def build_filename(coeffs: List[complex], l_list: List[int], 
                      p_list: List[int], waist: float, prefix: str = "LG_") -> str:
        """构建标准化文件名"""
        parts = []
        for c, p, l in zip(coeffs, p_list, l_list):
            amp = abs(c)
            phase_over_pi = np.angle(c) / np.pi
            parts.append(f"{amp:.3f},{phase_over_pi:.2f}pi({p},{l})")
        
        w_mm = waist * 1e3
        return f"{prefix}{'+'.join(parts)}_w_{w_mm:.3f}mm"
```

#### 3.1.2 开闭原则 (OCP) - 扩展开放，修改封闭
```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# 定义处理器接口
@runtime_checkable
class PhaseProcessor(Protocol):
    """相位处理器协议"""
    def process(self, phase_data: np.ndarray) -> np.ndarray:
        """处理相位数据"""
        ...

class StandardPhaseProcessor:
    """标准相位处理器"""
    
    def process(self, phase_data: np.ndarray) -> np.ndarray:
        """标准的相位取模和归一化"""
        # 取模到 [0, 2π]
        phase_normalized = np.mod(phase_data, 2 * np.pi)
        # 映射到 [0, 255]
        return (phase_normalized / (2 * np.pi) * 255).astype(np.uint8)

class EnhancedPhaseProcessor:
    """增强相位处理器 - 扩展功能而不修改原有代码"""
    
    def __init__(self, gamma: float = 1.0, noise_reduction: bool = False, 
                 edge_enhancement: bool = False):
        self.gamma = gamma
        self.noise_reduction = noise_reduction
        self.edge_enhancement = edge_enhancement
    
    def process(self, phase_data: np.ndarray) -> np.ndarray:
        """增强的相位处理"""
        phase_normalized = np.mod(phase_data, 2 * np.pi)
        
        # 增强功能1：伽马校正
        if self.gamma != 1.0:
            phase_normalized = np.power(
                phase_normalized / (2 * np.pi), self.gamma
            ) * (2 * np.pi)
        
        # 增强功能2：噪声抑制
        if self.noise_reduction:
            from scipy import ndimage
            phase_normalized = ndimage.gaussian_filter(phase_normalized, sigma=0.5)
        
        # 增强功能3：边缘增强
        if self.edge_enhancement:
            from scipy import ndimage
            edges = ndimage.laplace(phase_normalized)
            phase_normalized = phase_normalized + 0.1 * edges
        
        # 重新取模并映射
        phase_normalized = np.mod(phase_normalized, 2 * np.pi)
        return (phase_normalized / (2 * np.pi) * 255).astype(np.uint8)

class CustomPhaseProcessor:
    """自定义相位处理器 - 演示扩展性"""
    
    def __init__(self, custom_function: callable = None):
        self.custom_function = custom_function or (lambda x: x)
    
    def process(self, phase_data: np.ndarray) -> np.ndarray:
        """使用自定义函数处理相位"""
        processed = self.custom_function(phase_data)
        phase_normalized = np.mod(processed, 2 * np.pi)
        return (phase_normalized / (2 * np.pi) * 255).astype(np.uint8)

# 使用策略模式的生成器
class AdvancedLGGenerator:
    """支持不同相位处理策略的生成器"""
    
    def __init__(self, config: LGConfig, phase_processor: PhaseProcessor = None):
        self.config = config
        self.phase_processor = phase_processor or StandardPhaseProcessor()
    
    def set_phase_processor(self, processor: PhaseProcessor) -> None:
        """运行时切换处理策略"""
        if not isinstance(processor, PhaseProcessor):
            raise TypeError("处理器必须实现PhaseProcessor协议")
        self.phase_processor = processor
    
    def generate_with_processing(self, coeffs: List[complex], l_list: List[int], 
                               p_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """使用可配置的相位处理生成全息图"""
        # 计算原始相位
        raw_phase = self._compute_raw_phase(coeffs, l_list, p_list)
        
        # 使用策略处理相位
        processed_phase = self.phase_processor.process(raw_phase)
        
        # 计算全息图
        hologram = self._compute_hologram(raw_phase)
        
        return processed_phase, hologram
    
    def _compute_raw_phase(self, coeffs, l_list, p_list) -> np.ndarray:
        """计算原始相位（示例实现）"""
        # 这里会有实际的相位计算逻辑
        H, V = self.config.H, self.config.V
        return np.random.random((V, H)) * 2 * np.pi  # 示例
    
    def _compute_hologram(self, phase: np.ndarray) -> np.ndarray:
        """计算全息图（示例实现）"""
        # 这里会有实际的全息图计算逻辑
        return (np.sin(phase) * 127 + 128).astype(np.uint8)  # 示例

# 使用示例
def demonstrate_ocp():
    """演示开闭原则的应用"""
    config = LGConfig()
    generator = AdvancedLGGenerator(config)
    
    # 使用标准处理器
    phase1, holo1 = generator.generate_with_processing([1+0j], [1], [0])
    
    # 切换到增强处理器
    enhanced_processor = EnhancedPhaseProcessor(gamma=1.2, noise_reduction=True)
    generator.set_phase_processor(enhanced_processor)
    phase2, holo2 = generator.generate_with_processing([1+0j], [1], [0])
    
    # 使用自定义处理器
    custom_processor = CustomPhaseProcessor(lambda x: x + np.sin(x))
    generator.set_phase_processor(custom_processor)
    phase3, holo3 = generator.generate_with_processing([1+0j], [1], [0])
    
    return (phase1, phase2, phase3), (holo1, holo2, holo3)
```

#### 3.1.3 依赖注入与控制反转
```python
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod

# 定义配置提供者协议
@runtime_checkable
class ConfigProvider(Protocol):
    """配置提供者协议"""
    def get_image_size(self) -> Tuple[int, int]:
        """获取图像尺寸"""
        ...
    
    def get_pixel_size(self) -> float:
        """获取像素大小"""
        ...
    
    def get_optical_params(self) -> Dict[str, Any]:
        """获取光学参数"""
        ...

class FileConfigProvider:
    """从文件读取配置的提供者"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'H': 1920, 'V': 1152,
            'pixel_size': 1.25e-5,
            'default_waist': 0.00254,
            'enable_grating': True
        }
    
    def get_image_size(self) -> Tuple[int, int]:
        return self._config_data.get('H', 1920), self._config_data.get('V', 1152)
    
    def get_pixel_size(self) -> float:
        return self._config_data.get('pixel_size', 1.25e-5)
    
    def get_optical_params(self) -> Dict[str, Any]:
        return {
            'default_waist': self._config_data.get('default_waist', 0.00254),
            'enable_grating': self._config_data.get('enable_grating', True)
        }

class EnvironmentConfigProvider:
    """从环境变量读取配置的提供者"""
    
    def get_image_size(self) -> Tuple[int, int]:
        import os
        H = int(os.getenv('OAM_IMAGE_WIDTH', '1920'))
        V = int(os.getenv('OAM_IMAGE_HEIGHT', '1152'))
        return H, V
    
    def get_pixel_size(self) -> float:
        import os
        return float(os.getenv('OAM_PIXEL_SIZE', '1.25e-5'))
    
    def get_optical_params(self) -> Dict[str, Any]:
        import os
        return {
            'default_waist': float(os.getenv('OAM_DEFAULT_WAIST', '0.00254')),
            'enable_grating': os.getenv('OAM_ENABLE_GRATING', 'true').lower() == 'true'
        }

class DatabaseConfigProvider:
    """从数据库读取配置的提供者（示例）"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_image_size(self) -> Tuple[int, int]:
        # 模拟数据库查询
        result = self.db.execute("SELECT H, V FROM image_config WHERE active=1")
        return result[0] if result else (1920, 1152)
    
    def get_pixel_size(self) -> float:
        result = self.db.execute("SELECT pixel_size FROM optical_config WHERE active=1")
        return result[0] if result else 1.25e-5
    
    def get_optical_params(self) -> Dict[str, Any]:
        result = self.db.execute("SELECT * FROM optical_config WHERE active=1")
        return result[0] if result else {'default_waist': 0.00254, 'enable_grating': True}

# 依赖注入的生成器
class DILGGenerator:
    """使用依赖注入的LG生成器"""
    
    def __init__(self, config_provider: ConfigProvider, 
                 logger: Optional[Any] = None,
                 file_manager: Optional[Any] = None):
        """
        通过依赖注入初始化生成器
        
        Args:
            config_provider: 配置提供者，实现ConfigProvider协议
            logger: 日志记录器（可选）
            file_manager: 文件管理器（可选）
        """
        self.config_provider = config_provider
        self.logger = logger or self._create_default_logger()
        self.file_manager = file_manager or FileManager()
        
        # 从配置提供者获取参数
        self.H, self.V = config_provider.get_image_size()
        self.pixel_size = config_provider.get_pixel_size()
        self.optical_params = config_provider.get_optical_params()
        
        self.logger.info(f"生成器初始化完成: {self.H}x{self.V}, {self.pixel_size}")
    
    def _create_default_logger(self):
        """创建默认日志记录器"""
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def generate_coordinate_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成坐标网格"""
        self.logger.debug("生成坐标网格")
        x = np.linspace(-self.H/2, self.H/2-1, self.H) * self.pixel_size
        y = np.linspace(-self.V/2, self.V/2-1, self.V) * self.pixel_size
        return np.meshgrid(x, y)
    
    def generate_and_save(self, coeffs: List[complex], l_list: List[int], 
                         p_list: List[int], output_path: Path) -> None:
        """生成并保存全息图"""
        try:
            self.logger.info(f"开始生成全息图: {len(coeffs)}个模式")
            
            # 生成图像（这里简化为示例）
            phase_map = np.random.randint(0, 256, (self.V, self.H), dtype=np.uint8)
            hologram = np.random.randint(0, 256, (self.V, self.H), dtype=np.uint8)
            
            # 保存文件
            filename = self.file_manager.build_filename(
                coeffs, l_list, p_list, self.optical_params['default_waist']
            )
            
            phase_path = output_path / f"{filename}_phase.bmp"
            holo_path = output_path / f"{filename}_holo.bmp"
            
            self.file_manager.save_image(phase_path, phase_map)
            self.file_manager.save_image(holo_path, hologram)
            
            self.logger.info(f"保存完成: {phase_path}, {holo_path}")
            
        except Exception as e:
            self.logger.error(f"生成失败: {e}")
            raise

# 工厂模式 + 依赖注入
class LGGeneratorFactory:
    """LG生成器工厂"""
    
    @staticmethod
    def create_from_file(config_file: Path, **kwargs) -> DILGGenerator:
        """从配置文件创建生成器"""
        config_provider = FileConfigProvider(config_file)
        return DILGGenerator(config_provider, **kwargs)
    
    @staticmethod
    def create_from_environment(**kwargs) -> DILGGenerator:
        """从环境变量创建生成器"""
        config_provider = EnvironmentConfigProvider()
        return DILGGenerator(config_provider, **kwargs)
    
    @staticmethod
    def create_from_database(db_connection, **kwargs) -> DILGGenerator:
        """从数据库创建生成器"""
        config_provider = DatabaseConfigProvider(db_connection)
        return DILGGenerator(config_provider, **kwargs)

# 使用示例
def demonstrate_dependency_injection():
    """演示依赖注入的使用"""
    
    # 方式1：从文件配置创建
    if Path("config.json").exists():
        generator1 = LGGeneratorFactory.create_from_file(Path("config.json"))
    
    # 方式2：从环境变量创建
    generator2 = LGGeneratorFactory.create_from_environment()
    
    # 方式3：手动注入依赖
    config_provider = FileConfigProvider(Path("custom_config.json"))
    
    import logging
    custom_logger = logging.getLogger("CustomLG")
    custom_logger.setLevel(logging.DEBUG)
    
    generator3 = DILGGenerator(
        config_provider=config_provider,
        logger=custom_logger,
        file_manager=FileManager()
    )
    
    return generator1, generator2, generator3
```

### 3.2 设计模式的实际应用

#### 3.2.1 工厂模式 - 创建不同类型的生成器
```python
from enum import Enum
from typing import Dict, Type, Any

class GeneratorType(Enum):
    """生成器类型枚举"""
    STANDARD = "standard"
    HIGH_PRECISION = "high_precision"
    FAST = "fast"
    GPU_ACCELERATED = "gpu"

class BaseGenerator(ABC):
    """生成器基类"""
    
    def __init__(self, config: LGConfig):
        self.config = config
    
    @abstractmethod
    def generate_single(self, coeffs: List[complex], l_list: List[int], 
                       p_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """生成单个全息图"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取生成器信息"""
        pass

class StandardGenerator(BaseGenerator):
    """标准精度生成器"""
    
    def generate_single(self, coeffs: List[complex], l_list: List[int], 
                       p_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """标准精度计算"""
        # 使用float64精度
        from generatePhase_G_direct import generatePhase_G_direct
        return generatePhase_G_direct(
            H=self.config.H, V=self.config.V,
            w=self.config.default_waist, wd=[],
            coeffs=coeffs, l_list=l_list, p_list=p_list,
            r=self.config.grating_weight, k=self.config.grating_period,
            nn=0, m=0, pixel_size=self.config.pixel_size
        )
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'Standard',
            'precision': 'float64',
            'speed': 'medium',
            'memory_usage': 'medium'
        }

class HighPrecisionGenerator(BaseGenerator):
    """高精度生成器"""
    
    def generate_single(self, coeffs: List[complex], l_list: List[int], 
                       p_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """高精度计算，使用更精确的数值方法"""
        # 使用更高精度的数值类型和更稳定的算法
        coeffs_hp = [np.complex128(c) for c in coeffs]
        
        # 这里会调用高精度版本的算法
        # 例如使用更多的递推项、更高的采样率等
        phase_map = np.zeros((self.config.V, self.config.H), dtype=np.float64)
        hologram = np.zeros((self.config.V, self.config.H), dtype=np.float64)
        
        # 高精度计算逻辑...
        # （这里简化为示例）
        
        return phase_map.astype(np.uint8), hologram.astype(np.uint8)
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'High Precision',
            'precision': 'float128',
            'speed': 'slow',
            'memory_usage': 'high'
        }

class FastGenerator(BaseGenerator):
    """快速生成器"""
    
    def __init__(self, config: LGConfig, approximation_level: int = 1):
        super().__init__(config)
        self.approximation_level = approximation_level
    
    def generate_single(self, coeffs: List[complex], l_list: List[int], 
                       p_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """快速计算，使用近似算法"""
        # 使用float32节省内存和提高速度
        # 使用近似的拉盖尔多项式计算
        # 降低采样率
        
        # 简化的快速算法
        H_fast = self.config.H // 2  # 降低分辨率
        V_fast = self.config.V // 2
        
        phase_map_fast = np.random.random((V_fast, H_fast)).astype(np.float32)
        hologram_fast = np.random.random((V_fast, H_fast)).astype(np.float32)
        
        # 插值回原分辨率
        from scipy import ndimage
        phase_map = ndimage.zoom(phase_map_fast, 2, order=1)
        hologram = ndimage.zoom(hologram_fast, 2, order=1)
        
        return (phase_map * 255).astype(np.uint8), (hologram * 255).astype(np.uint8)
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'Fast',
            'precision': 'float32',
            'speed': 'fast',
            'memory_usage': 'low',
            'approximation_level': self.approximation_level
        }

class GPUGenerator(BaseGenerator):
    """GPU加速生成器（示例）"""
    
    def __init__(self, config: LGConfig):
        super().__init__(config)
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        try:
            import cupy as cp
            self.gpu_available = True
            self.cp = cp
        except ImportError:
            self.gpu_available = False
            print("警告: CuPy未安装，GPU加速不可用")
    
    def generate_single(self, coeffs: List[complex], l_list: List[int], 
                       p_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """GPU加速计算"""
        if not self.gpu_available:
            # 回退到CPU计算
            fallback = StandardGenerator(self.config)
            return fallback.generate_single(coeffs, l_list, p_list)
        
        # GPU计算逻辑
        cp = self.cp
        
        # 将数据传输到GPU
        coeffs_gpu = cp.array(coeffs)
        
        # GPU上的并行计算
        # （这里简化为示例）
        phase_map_gpu = cp.random.random((self.config.V, self.config.H))
        hologram_gpu = cp.random.random((self.config.V, self.config.H))
        
        # 传回CPU
        phase_map = cp.asnumpy(phase_map_gpu * 255).astype(np.uint8)
        hologram = cp.asnumpy(hologram_gpu * 255).astype(np.uint8)
        
        return phase_map, hologram
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'GPU Accelerated',
            'precision': 'float32',
            'speed': 'very_fast' if self.gpu_available else 'medium',
            'memory_usage': 'gpu' if self.gpu_available else 'medium',
            'gpu_available': self.gpu_available
        }

class GeneratorFactory:
    """生成器工厂类"""
    
    _generators: Dict[GeneratorType, Type[BaseGenerator]] = {
        GeneratorType.STANDARD: StandardGenerator,
        GeneratorType.HIGH_PRECISION: HighPrecisionGenerator,
        GeneratorType.FAST: FastGenerator,
        GeneratorType.GPU_ACCELERATED: GPUGenerator,
    }
    
    @classmethod
    def create_generator(cls, generator_type: GeneratorType, 
                        config: LGConfig, **kwargs) -> BaseGenerator:
        """创建指定类型的生成器"""
        if generator_type not in cls._generators:
            raise ValueError(f"不支持的生成器类型: {generator_type}")
        
        generator_class = cls._generators[generator_type]
        return generator_class(config, **kwargs)
    
    @classmethod
    def register_generator(cls, generator_type: GeneratorType, 
                          generator_class: Type[BaseGenerator]) -> None:
        """注册新的生成器类型"""
        cls._generators[generator_type] = generator_class
    
    @classmethod
    def list_available_generators(cls) -> List[Dict[str, Any]]:
        """列出所有可用的生成器"""
        result = []
        config = LGConfig()  # 临时配置用于获取信息
        
        for gen_type, gen_class in cls._generators.items():
            try:
                generator = gen_class(config)
                info = generator.get_info()
                info['enum_type'] = gen_type
                result.append(info)
            except Exception as e:
                result.append({
                    'enum_type': gen_type,
                    'type': gen_class.__name__,
                    'error': str(e),
                    'available': False
                })
        
        return result
    
    @classmethod
    def auto_select_generator(cls, config: LGConfig, 
                            priority: str = "balanced") -> BaseGenerator:
        """自动选择最适合的生成器"""
        image_size = config.H * config.V
        
        if priority == "speed":
            if image_size > 1920 * 1152:
                return cls.create_generator(GeneratorType.GPU_ACCELERATED, config)
            else:
                return cls.create_generator(GeneratorType.FAST, config)
        
        elif priority == "precision":
            return cls.create_generator(GeneratorType.HIGH_PRECISION, config)
        
        elif priority == "balanced":
            if image_size > 2048 * 2048:
                return cls.create_generator(GeneratorType.GPU_ACCELERATED, config)
            else:
                return cls.create_generator(GeneratorType.STANDARD, config)
        
        else:
            raise ValueError(f"未知的优先级: {priority}")

# 使用示例
def demonstrate_factory_pattern():
    """演示工厂模式的使用"""
    config = LGConfig()
    
    # 创建不同类型的生成器
    generators = {}
    
    for gen_type in GeneratorType:
        try:
            generator = GeneratorFactory.create_generator(gen_type, config)
            generators[gen_type.value] = generator
            print(f"✓ 创建 {gen_type.value} 生成器成功")
        except Exception as e:
            print(f"✗ 创建 {gen_type.value} 生成器失败: {e}")
    
    # 列出可用生成器
    available = GeneratorFactory.list_available_generators()
    print("\n可用生成器:")
    for info in available:
        print(f"  {info['type']}: 速度={info.get('speed', 'unknown')}, "
              f"精度={info.get('precision', 'unknown')}")
    
    # 自动选择
    auto_gen = GeneratorFactory.auto_select_generator(config, priority="speed")
    print(f"\n自动选择的生成器: {auto_gen.__class__.__name__}")
    
    return generators, available, auto_gen
```

---

## 第4章：NumPy 数组计算与数学建模

### 4.1 NumPy 基础与数组创建

#### 4.1.1 数组创建的多种方式及在项目中的应用
```python
import numpy as np
from typing import Tuple, Dict, Optional
import time

def array_creation_comprehensive():
    """全面的数组创建方法"""
    
    # 1. 基础创建方法 - 项目核心需求
    def create_coordinate_arrays(H: int, V: int, pixel_size: float) -> Dict[str, np.ndarray]:
        """创建坐标数组 - OAM项目的核心需求"""
        
        # 线性空间创建（最常用）
        x = np.linspace(-H/2, H/2-1, H, dtype=np.float64) * pixel_size
        y = np.linspace(-V/2, V/2-1, V) * pixel_size
        
        # 等价的其他创建方式对比
        x_arange = np.arange(-H/2, H/2, dtype=np.float64) * pixel_size
        y_arange = np.arange(-V/2, V/2) * pixel_size
        
        # 网格生成（重要：indexing参数的影响）
        X_xy, Y_xy = np.meshgrid(x, y, indexing='xy')  # Cartesian indexing
        X_ij, Y_ij = np.meshgrid(x, y, indexing='ij')  # Matrix indexing
        
        print(f"XY索引: X形状={X_xy.shape}, Y形状={Y_xy.shape}")
        print(f"IJ索引: X形状={X_ij.shape}, Y形状={Y_ij.shape}")
        
        # 极坐标计算
        rho = np.sqrt(X_xy**2 + Y_xy**2)
        phi = np.angle(X_xy + 1j * Y_xy)  # 使用复数计算角度
        
        return {
            'x_linear': x, 'y_linear': y,
            'x_arange': x_arange, 'y_arange': y_arange,
            'X_cartesian': X_xy, 'Y_cartesian': Y_xy,
            'X_matrix': X_ij, 'Y_matrix': Y_ij,
            'rho': rho, 'phi': phi
        }
    
    # 2. 特殊数组创建
    def initialize_computation_arrays(shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """初始化计算用数组"""
        V, H = shape
        
        arrays = {
            # 基础数组
            'zeros_real': np.zeros((V, H), dtype=np.float64),        # 实数零数组
            'zeros_complex': np.zeros((V, H), dtype=np.complex128),  # 复数零数组
            'ones': np.ones((V, H), dtype=np.float64),               # 全1数组
            'empty': np.empty((V, H), dtype=np.float64),             # 未初始化（更快）
            
            # 特殊值数组
            'nan_array': np.full((V, H), np.nan),                    # NaN数组
            'inf_array': np.full((V, H), np.inf),                    # 无穷大数组
            'pi_array': np.full((V, H), np.pi),                      # π数组
            
            # 单位矩阵和对角矩阵
            'identity': np.eye(min(V, H)),                           # 单位矩阵
            'diagonal': np.diag(np.arange(min(V, H))),               # 对角矩阵
            
            # 随机数组
            'random_uniform': np.random.random((V, H)),              # [0,1)均匀分布
            'random_normal': np.random.normal(0, 1, (V, H)),        # 标准正态分布
            'random_phase': np.random.random((V, H)) * 2 * np.pi,   # [0,2π)相位
            
            # 图像数组
            'uint8_image': np.zeros((V, H), dtype=np.uint8),         # 8位图像
            'uint16_image': np.zeros((V, H), dtype=np.uint16),       # 16位图像
        }
        
        return arrays
    
    # 3. 从函数创建数组
    def create_analytical_patterns(H: int, V: int) -> Dict[str, np.ndarray]:
        """创建解析函数图案"""
        
        # 使用 fromfunction 创建复杂图案
        def gaussian_pattern(i, j):
            """高斯图案"""
            center_i, center_j = V//2, H//2
            sigma = min(V, H) / 8
            return np.exp(-((i - center_i)**2 + (j - center_j)**2) / (2 * sigma**2))
        
        def sine_wave_pattern(i, j):
            """正弦波图案"""
            freq_i, freq_j = 2*np.pi/V, 2*np.pi/H
            return np.sin(freq_i * i) * np.cos(freq_j * j)
        
        def checkerboard_pattern(i, j):
            """棋盘图案"""
            return ((i // 20) + (j // 20)) % 2
        
        def radial_pattern(i, j):
            """径向图案"""
            center_i, center_j = V//2, H//2
            r = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            return np.sin(r / 10) * np.exp(-r / 100)
        
        patterns = {
            'gaussian': np.fromfunction(gaussian_pattern, (V, H)),
            'sine_wave': np.fromfunction(sine_wave_pattern, (V, H)),
            'checkerboard': np.fromfunction(checkerboard_pattern, (V, H)),
            'radial': np.fromfunction(radial_pattern, (V, H)),
            'linear_gradient': np.fromfunction(lambda i, j: j / H, (V, H))
        }
        
        return patterns
    
    # 4. 高级创建技巧
    def advanced_creation_techniques():
        """高级数组创建技巧"""
        
        # 使用ogrid和mgrid
        def demonstrate_grid_differences():
            """演示ogrid和mgrid的区别"""
            # ogrid: 开放网格（节省内存）
            y_og, x_og = np.ogrid[0:100, 0:200]
            print(f"ogrid形状: y={y_og.shape}, x={x_og.shape}")
            
            # mgrid: 完整网格
            y_mg, x_mg = np.mgrid[0:100, 0:200]
            print(f"mgrid形状: y={y_mg.shape}, x={x_mg.shape}")
            
            # 内存使用对比
            print(f"ogrid内存: {y_og.nbytes + x_og.nbytes} bytes")
            print(f"mgrid内存: {y_mg.nbytes + x_mg.nbytes} bytes")
            
            return (y_og, x_og), (y_mg, x_mg)
        
        # 结构化数组
        def create_structured_arrays():
            """创建结构化数组"""
            # 定义复合数据类型
            mode_dtype = np.dtype([
                ('l', 'i4'),           # 角量子数
                ('p', 'i4'),           # 径向量子数  
                ('amplitude', 'f8'),    # 幅度
                ('phase', 'f8'),       # 相位
                ('active', '?')        # 是否激活
            ])
            
            # 创建结构化数组
            modes = np.zeros(10, dtype=mode_dtype)
            modes['l'] = np.arange(-5, 5)
            modes['p'] = np.abs(modes['l']) // 2
            modes['amplitude'] = np.random.random(10)
            modes['phase'] = np.random.random(10) * 2 * np.pi
            modes['active'] = modes['amplitude'] > 0.5
            
            return modes
        
        # 记录数组（类似结构化数组但更灵活）
        def create_record_arrays():
            """创建记录数组"""
            # 使用字段名直接访问
            data = np.rec.fromarrays([
                np.arange(5),                    # l值
                np.zeros(5),                     # p值
                np.random.random(5),             # 幅度
                np.random.random(5) * 2 * np.pi  # 相位
            ], names='l,p,amplitude,phase')
            
            return data
        
        return {
            'grid_demo': demonstrate_grid_differences(),
            'structured': create_structured_arrays(),
            'records': create_record_arrays()
        }
    
    # 测试示例
    H, V = 256, 256
    results = {
        'coordinates': create_coordinate_arrays(H, V, 1.25e-5),
        'basic_arrays': initialize_computation_arrays((V, H)),
        'patterns': create_analytical_patterns(H, V),
        'advanced': advanced_creation_techniques()
    }
    
    return results
```

#### 4.1.2 数组属性深入分析与内存管理
```python
def array_properties_and_memory():
    """数组属性分析与内存管理"""
    
    def analyze_array_properties(arr: np.ndarray) -> Dict[str, Any]:
        """详细分析数组属性"""
        analysis = {
            # 基础属性
            'shape': arr.shape,
            'ndim': arr.ndim,
            'size': arr.size,
            'dtype': str(arr.dtype),
            'itemsize': arr.itemsize,
            'nbytes': arr.nbytes,
            'nbytes_mb': arr.nbytes / 1024**2,
            
            # 内存布局
            'strides': arr.strides,
            'flags': {
                'C_CONTIGUOUS': arr.flags['C_CONTIGUOUS'],
                'F_CONTIGUOUS': arr.flags['F_CONTIGUOUS'],
                'OWNDATA': arr.flags['OWNDATA'],
                'WRITEABLE': arr.flags['WRITEABLE'],
                'ALIGNED': arr.flags['ALIGNED'],
                'WRITEBACKIFCOPY': arr.flags['WRITEBACKIFCOPY']
            },
            
            # 数据统计
            'min': float(arr.min()) if arr.size > 0 else None,
            'max': float(arr.max()) if arr.size > 0 else None,
            'mean': float(arr.mean()) if arr.size > 0 else None,
            'std': float(arr.std()) if arr.size > 0 else None,
        }
        
        return analysis
    
    def memory_layout_optimization():
        """内存布局优化"""
        
        # 创建测试数组
        large_array = np.random.random((1000, 1000))
        
        # C风格 vs Fortran风格
        c_style = np.array(large_array, order='C')
        f_style = np.array(large_array, order='F')
        
        print("内存布局对比:")
        print(f"C风格连续: {c_style.flags['C_CONTIGUOUS']}")
        print(f"F风格连续: {f_style.flags['F_CONTIGUOUS']}")
        print(f"C风格步长: {c_style.strides}")
        print(f"F风格步长: {f_style.strides}")
        
        # 性能测试：按行访问 vs 按列访问
        def benchmark_access_patterns():
            """基准测试：访问模式对性能的影响"""
            import time
            
            # 按行访问（C风格友好）
            start = time.perf_counter()
            for i in range(c_style.shape[0]):
                row_sum = c_style[i, :].sum()
            c_row_time = time.perf_counter() - start
            
            # 按列访问（F风格友好）
            start = time.perf_counter()
            for j in range(c_style.shape[1]):
                col_sum = c_style[:, j].sum()
            c_col_time = time.perf_counter() - start
            
            # F风格数组的相同测试
            start = time.perf_counter()
            for i in range(f_style.shape[0]):
                row_sum = f_style[i, :].sum()
            f_row_time = time.perf_counter() - start
            
            start = time.perf_counter()
            for j in range(f_style.shape[1]):
                col_sum = f_style[:, j].sum()
            f_col_time = time.perf_counter() - start
            
            return {
                'c_array_row_access': c_row_time,
                'c_array_col_access': c_col_time,
                'f_array_row_access': f_row_time,
                'f_array_col_access': f_col_time
            }
        
        return benchmark_access_patterns()
    
    def memory_optimization_techniques():
        """内存优化技术"""
        
        # 1. 数据类型优化
        def dtype_optimization():
            """数据类型优化"""
            # 原始数据（float64）
            original = np.random.random((2000, 2000))
            
            # 优化后的数据类型
            optimized_f32 = original.astype(np.float32)
            optimized_f16 = original.astype(np.float16)
            
            # 内存使用对比
            memory_comparison = {
                'float64_mb': original.nbytes / 1024**2,
                'float32_mb': optimized_f32.nbytes / 1024**2,
                'float16_mb': optimized_f16.nbytes / 1024**2
            }
            
            # 精度损失分析
            precision_loss = {
                'f32_max_error': np.max(np.abs(original - optimized_f32.astype(np.float64))),
                'f16_max_error': np.max(np.abs(original - optimized_f16.astype(np.float64))),
                'f32_mean_error': np.mean(np.abs(original - optimized_f32.astype(np.float64))),
                'f16_mean_error': np.mean(np.abs(original - optimized_f16.astype(np.float64)))
            }
            
            return memory_comparison, precision_loss
        
        # 2. 视图 vs 副本
        def view_vs_copy_analysis():
            """视图与副本分析"""
            original = np.random.random((1000, 1000))
            
            # 创建视图（共享内存）
            view_slice = original[::2, ::2]      # 切片创建视图
            view_transpose = original.T          # 转置创建视图
            view_reshape = original.reshape(-1)  # 重塑（如果可能）创建视图
            
            # 创建副本（独立内存）
            copy_explicit = original.copy()                    # 显式复制
            copy_slice = original[original > 0.5]             # 布尔索引创建副本
            copy_fancy = original[[0, 2, 4], :]               # 花式索引创建副本
            
            # 内存共享检查
            shares_memory = {
                'slice_view': np.shares_memory(original, view_slice),
                'transpose_view': np.shares_memory(original, view_transpose),
                'reshape_view': np.shares_memory(original, view_reshape),
                'explicit_copy': np.shares_memory(original, copy_explicit),
                'bool_index_copy': np.shares_memory(original, copy_slice),
                'fancy_index_copy': np.shares_memory(original, copy_fancy)
            }
            
            return shares_memory
        
        # 3. 原地操作
        def inplace_operations():
            """原地操作示例"""
            data = np.random.random((1000, 1000))
            original_id = id(data)
            
            # 原地操作（推荐）
            data += 1              # 原地加法
            data *= 2              # 原地乘法
            data **= 0.5           # 原地幂运算
            np.clip(data, 0, 1, out=data)  # 原地裁剪
            
            # 验证仍然是原始数组
            same_object = id(data) == original_id
            
            # 非原地操作对比
            data2 = np.random.random((1000, 1000))
            original_id2 = id(data2)
            
            data2 = data2 + 1      # 创建新数组
            data2 = data2 * 2      # 创建新数组
            
            new_object = id(data2) != original_id2
            
            return {
                'inplace_preserves_object': same_object,
                'new_operations_create_object': new_object
            }
        
        return {
            'dtype_optimization': dtype_optimization(),
            'view_copy_analysis': view_vs_copy_analysis(),
            'inplace_operations': inplace_operations()
        }
    
    def safe_dtype_conversion():
        """安全的数据类型转换"""
        
        def convert_with_validation(arr: np.ndarray, target_dtype: np.dtype) -> Tuple[np.ndarray, Dict]:
            """带验证的数据类型转换"""
            conversion_info = {
                'original_dtype': str(arr.dtype),
                'target_dtype': str(target_dtype),
                'conversion_safe': True,
                'warnings': []
            }
            
            if arr.dtype == target_dtype:
                return arr, conversion_info
            
            # 检查值域是否适合目标类型
            if target_dtype == np.uint8:
                arr_min, arr_max = arr.min(), arr.max()
                if arr_min < 0:
                    conversion_info['warnings'].append(f"数据包含负值 {arr_min}")
                    conversion_info['conversion_safe'] = False
                if arr_max > 255:
                    conversion_info['warnings'].append(f"数据包含超出255的值 {arr_max}")
                    conversion_info['conversion_safe'] = False
                
                # 安全转换
                if not conversion_info['conversion_safe']:
                    arr_clipped = np.clip(arr, 0, 255)
                    result = arr_clipped.astype(target_dtype)
                else:
                    result = arr.astype(target_dtype)
            
            elif target_dtype == np.float32:
                # 检查精度损失
                test_conversion = arr.astype(target_dtype).astype(arr.dtype)
                max_error = np.max(np.abs(arr - test_conversion))
                if max_error > 1e-6:
                    conversion_info['warnings'].append(f"精度损失: 最大误差 {max_error}")
                
                result = arr.astype(target_dtype)
            
            else:
                result = arr.astype(target_dtype)
            
            return result, conversion_info
        
        # 测试不同的转换场景
        test_arrays = {
            'normal_range': np.random.random((100, 100)) * 255,
            'negative_values': np.random.normal(0, 50, (100, 100)),
            'large_values': np.random.random((100, 100)) * 1000,
            'high_precision': np.random.random((100, 100)) * 1e-10
        }
        
        conversion_results = {}
        for name, arr in test_arrays.items():
            result, info = convert_with_validation(arr, np.uint8)
            conversion_results[name] = info
        
        return conversion_results
    
    # 运行测试
    test_array = np.random.random((500, 500))
    
    return {
        'array_analysis': analyze_array_properties(test_array),
        'memory_layout': memory_layout_optimization(),
        'memory_optimization': memory_optimization_techniques(),
        'safe_conversion': safe_dtype_conversion()
    }
```

---

**文档状态**: 第3-4章详细内容已完成 ✅

**包含内容**:
- 面向对象设计原则（SRP、OCP、DIP）
- 设计模式实际应用（工厂模式、策略模式等）
- 依赖注入与控制反转
- NumPy数组创建的各种方法
- 数组属性分析与内存优化

**下一步**: 准备创建第5-6章（SciPy科学计算与数学算法实现）

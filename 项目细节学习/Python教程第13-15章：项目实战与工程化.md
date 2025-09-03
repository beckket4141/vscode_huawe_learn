# Python 教程第13-15章：项目实战与工程化

> **本文档涵盖**：第13章 项目架构设计与模块化、第14章 错误处理与调试技巧、第15章 性能优化与测试策略

---

## 第13章：项目架构设计与模块化

### 13.1 分层架构设计实践

#### 13.1.1 OAM项目架构深度解析
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue

def project_architecture_design():
    """项目架构设计完整实现"""
    
    # ==================== 基础层 (Foundation Layer) ====================
    
    class ConfigurationError(Exception):
        """配置错误"""
        pass
    
    class ProcessingError(Exception):
        """处理错误"""
        pass
    
    class ValidationError(Exception):
        """验证错误"""
        pass
    
    @dataclass
    class SystemConfig:
        """系统配置"""
        image_width: int = 1920
        image_height: int = 1152
        pixel_size: float = 1.25e-5
        max_workers: int = 4
        temp_dir: str = "temp"
        log_level: str = "INFO"
        
        def validate(self) -> List[str]:
            """验证配置"""
            errors = []
            if self.image_width <= 0:
                errors.append("图像宽度必须为正数")
            if self.image_height <= 0:
                errors.append("图像高度必须为正数")
            if self.pixel_size <= 0:
                errors.append("像素尺寸必须为正数")
            if self.max_workers < 1:
                errors.append("工作线程数必须大于0")
            return errors
    
    @dataclass
    class OpticalConfig:
        """光学配置"""
        default_waist: float = 0.00254
        enable_waist_correction: bool = True
        wavelength: float = 632.8e-9
        
        enable_grating: bool = True
        grating_weight: float = -1.0
        grating_period: float = 12.0
        
        def validate(self) -> List[str]:
            """验证光学配置"""
            errors = []
            if self.default_waist <= 0:
                errors.append("默认光腰必须为正数")
            if self.wavelength <= 0:
                errors.append("波长必须为正数")
            if self.grating_period <= 0:
                errors.append("光栅周期必须为正数")
            return errors
    
    @dataclass
    class ModeParameters:
        """模式参数"""
        l: int
        p: int
        amplitude: float
        phase: float
        waist: Optional[float] = None
        enabled: bool = True
        
        def validate(self) -> List[str]:
            """验证模式参数"""
            errors = []
            if not isinstance(self.l, int):
                errors.append("l值必须为整数")
            if not isinstance(self.p, int) or self.p < 0:
                errors.append("p值必须为非负整数")
            if self.amplitude < 0:
                errors.append("幅度必须为非负数")
            if self.waist is not None and self.waist <= 0:
                errors.append("光腰必须为正数")
            return errors
    
    # ==================== 接口定义层 ====================
    
    @runtime_checkable
    class ConfigManager(Protocol):
        """配置管理器接口"""
        def load_config(self, path: Path) -> Dict[str, Any]:
            """加载配置"""
            ...
        
        def save_config(self, config: Dict[str, Any], path: Path) -> None:
            """保存配置"""
            ...
        
        def get_system_config(self) -> SystemConfig:
            """获取系统配置"""
            ...
        
        def get_optical_config(self) -> OpticalConfig:
            """获取光学配置"""
            ...
    
    @runtime_checkable
    class AlgorithmEngine(Protocol):
        """算法引擎接口"""
        def generate_phase_map(self, params: ModeParameters, 
                             optical_config: OpticalConfig,
                             system_config: SystemConfig) -> Any:
            """生成相位图"""
            ...
        
        def generate_hologram(self, params: ModeParameters,
                            optical_config: OpticalConfig,
                            system_config: SystemConfig) -> Any:
            """生成全息图"""
            ...
    
    @runtime_checkable
    class DataProcessor(Protocol):
        """数据处理器接口"""
        def process_batch(self, parameters: List[ModeParameters]) -> List[Any]:
            """批量处理"""
            ...
        
        def validate_parameters(self, parameters: List[ModeParameters]) -> List[str]:
            """验证参数"""
            ...
    
    @runtime_checkable
    class FileManager(Protocol):
        """文件管理器接口"""
        def save_image(self, image_data: Any, file_path: Path) -> None:
            """保存图像"""
            ...
        
        def load_parameters(self, file_path: Path) -> List[ModeParameters]:
            """加载参数"""
            ...
        
        def save_results(self, results: List[Any], output_dir: Path) -> None:
            """保存结果"""
            ...
    
    # ==================== 实现层 (Implementation Layer) ====================
    
    class JSONConfigManager:
        """JSON配置管理器实现"""
        
        def __init__(self):
            self._system_config = SystemConfig()
            self._optical_config = OpticalConfig()
            self._logger = logging.getLogger(__name__)
        
        def load_config(self, path: Path) -> Dict[str, Any]:
            """加载JSON配置"""
            try:
                if not path.exists():
                    self._logger.warning(f"配置文件不存在: {path}")
                    return {}
                
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 更新系统配置
                if 'system' in config_data:
                    system_data = config_data['system']
                    self._system_config = SystemConfig(**system_data)
                
                # 更新光学配置
                if 'optical' in config_data:
                    optical_data = config_data['optical']
                    self._optical_config = OpticalConfig(**optical_data)
                
                # 验证配置
                self._validate_configurations()
                
                self._logger.info(f"配置加载成功: {path}")
                return config_data
                
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"JSON格式错误: {e}")
            except Exception as e:
                raise ConfigurationError(f"加载配置失败: {e}")
        
        def save_config(self, config: Dict[str, Any], path: Path) -> None:
            """保存JSON配置"""
            try:
                # 确保目录存在
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # 准备配置数据
                config_data = {
                    'system': asdict(self._system_config),
                    'optical': asdict(self._optical_config),
                    **config
                }
                
                # 保存到文件
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                self._logger.info(f"配置保存成功: {path}")
                
            except Exception as e:
                raise ConfigurationError(f"保存配置失败: {e}")
        
        def get_system_config(self) -> SystemConfig:
            """获取系统配置"""
            return self._system_config
        
        def get_optical_config(self) -> OpticalConfig:
            """获取光学配置"""
            return self._optical_config
        
        def _validate_configurations(self):
            """验证所有配置"""
            errors = []
            errors.extend(self._system_config.validate())
            errors.extend(self._optical_config.validate())
            
            if errors:
                raise ConfigurationError(f"配置验证失败: {'; '.join(errors)}")
    
    class LGAlgorithmEngine:
        """LG算法引擎实现"""
        
        def __init__(self):
            self._logger = logging.getLogger(__name__)
            self._cache = {}
        
        def generate_phase_map(self, params: ModeParameters, 
                             optical_config: OpticalConfig,
                             system_config: SystemConfig) -> Any:
            """生成相位图"""
            try:
                # 验证参数
                errors = params.validate()
                if errors:
                    raise ValidationError(f"参数验证失败: {'; '.join(errors)}")
                
                # 检查缓存
                cache_key = self._generate_cache_key(params, optical_config, system_config)
                if cache_key in self._cache:
                    self._logger.debug(f"使用缓存结果: {cache_key}")
                    return self._cache[cache_key]['phase']
                
                # 实际生成（这里调用底层算法）
                phase_map = self._compute_phase_map(params, optical_config, system_config)
                
                # 更新缓存
                self._cache[cache_key] = {'phase': phase_map}
                
                self._logger.info(f"相位图生成完成: l={params.l}, p={params.p}")
                return phase_map
                
            except Exception as e:
                self._logger.error(f"相位图生成失败: {e}")
                raise ProcessingError(f"相位图生成失败: {e}")
        
        def generate_hologram(self, params: ModeParameters,
                            optical_config: OpticalConfig,
                            system_config: SystemConfig) -> Any:
            """生成全息图"""
            try:
                # 先生成相位图
                phase_map = self.generate_phase_map(params, optical_config, system_config)
                
                # 生成全息图
                hologram = self._compute_hologram(phase_map, optical_config)
                
                self._logger.info(f"全息图生成完成: l={params.l}, p={params.p}")
                return hologram
                
            except Exception as e:
                self._logger.error(f"全息图生成失败: {e}")
                raise ProcessingError(f"全息图生成失败: {e}")
        
        def _compute_phase_map(self, params: ModeParameters, 
                             optical_config: OpticalConfig,
                             system_config: SystemConfig) -> Any:
            """计算相位图（调用实际算法）"""
            import numpy as np
            
            # 这里会调用实际的generatePhase_G_direct函数
            # 为了演示，我们创建模拟数据
            H, V = system_config.image_width, system_config.image_height
            
            # 模拟LG模式相位计算
            x = np.linspace(-H/2, H/2-1, H) * system_config.pixel_size
            y = np.linspace(-V/2, V/2-1, V) * system_config.pixel_size
            X, Y = np.meshgrid(x, y)
            
            rho = np.sqrt(X**2 + Y**2)
            phi = np.angle(X + 1j*Y)
            
            # 简化的LG模式相位
            phase = params.l * phi + params.phase
            
            return phase
        
        def _compute_hologram(self, phase_map: Any, optical_config: OpticalConfig) -> Any:
            """计算全息图"""
            import numpy as np
            
            # 添加线性光栅（如果启用）
            if optical_config.enable_grating:
                H, V = phase_map.shape[1], phase_map.shape[0]
                x = np.arange(H)
                grating_phase = 2 * np.pi * x / optical_config.grating_period
                grating = optical_config.grating_weight * np.sin(grating_phase)
                phase_map = phase_map + grating[np.newaxis, :]
            
            # 生成全息图
            hologram = (np.sin(phase_map) + 1) * 127.5
            
            return hologram.astype(np.uint8)
        
        def _generate_cache_key(self, params: ModeParameters,
                              optical_config: OpticalConfig,
                              system_config: SystemConfig) -> str:
            """生成缓存键"""
            import hashlib
            
            key_data = {
                'l': params.l,
                'p': params.p,
                'amplitude': params.amplitude,
                'phase': params.phase,
                'waist': params.waist or optical_config.default_waist,
                'width': system_config.image_width,
                'height': system_config.image_height,
                'pixel_size': system_config.pixel_size
            }
            
            key_str = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    class BatchDataProcessor:
        """批量数据处理器实现"""
        
        def __init__(self, algorithm_engine: AlgorithmEngine,
                     config_manager: ConfigManager,
                     max_workers: int = 4):
            self.algorithm_engine = algorithm_engine
            self.config_manager = config_manager
            self.max_workers = max_workers
            self._logger = logging.getLogger(__name__)
        
        def process_batch(self, parameters: List[ModeParameters]) -> List[Dict[str, Any]]:
            """批量处理"""
            # 验证参数
            validation_errors = self.validate_parameters(parameters)
            if validation_errors:
                raise ValidationError(f"批量参数验证失败: {validation_errors}")
            
            # 获取配置
            system_config = self.config_manager.get_system_config()
            optical_config = self.config_manager.get_optical_config()
            
            results = []
            
            # 过滤启用的参数
            enabled_params = [p for p in parameters if p.enabled]
            self._logger.info(f"开始批量处理: {len(enabled_params)} 个任务")
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_param = {
                    executor.submit(self._process_single, param, optical_config, system_config): param
                    for param in enabled_params
                }
                
                # 收集结果
                for future in as_completed(future_to_param):
                    param = future_to_param[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self._logger.info(f"处理完成: l={param.l}, p={param.p}")
                    except Exception as e:
                        self._logger.error(f"处理失败: l={param.l}, p={param.p} - {e}")
                        results.append({
                            'parameters': param,
                            'success': False,
                            'error': str(e)
                        })
            
            self._logger.info(f"批量处理完成: {len(results)} 个结果")
            return results
        
        def validate_parameters(self, parameters: List[ModeParameters]) -> List[str]:
            """验证参数列表"""
            all_errors = []
            
            for i, param in enumerate(parameters):
                errors = param.validate()
                if errors:
                    all_errors.extend([f"第{i+1}组参数: {error}" for error in errors])
            
            return all_errors
        
        def _process_single(self, param: ModeParameters,
                          optical_config: OpticalConfig,
                          system_config: SystemConfig) -> Dict[str, Any]:
            """处理单个参数"""
            try:
                # 生成相位图和全息图
                phase_map = self.algorithm_engine.generate_phase_map(
                    param, optical_config, system_config
                )
                hologram = self.algorithm_engine.generate_hologram(
                    param, optical_config, system_config
                )
                
                return {
                    'parameters': param,
                    'phase_map': phase_map,
                    'hologram': hologram,
                    'success': True
                }
                
            except Exception as e:
                raise ProcessingError(f"处理失败: {e}")
    
    # ==================== 业务逻辑层 (Business Layer) ====================
    
    class OAMApplicationService:
        """OAM应用服务 - 业务逻辑层"""
        
        def __init__(self):
            # 依赖注入
            self.config_manager: ConfigManager = JSONConfigManager()
            self.algorithm_engine: AlgorithmEngine = LGAlgorithmEngine()
            self.data_processor: DataProcessor = BatchDataProcessor(
                self.algorithm_engine, self.config_manager
            )
            
            self._logger = logging.getLogger(__name__)
            self._setup_logging()
        
        def _setup_logging(self):
            """设置日志"""
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        def initialize_application(self, config_path: Optional[Path] = None):
            """初始化应用"""
            try:
                if config_path and config_path.exists():
                    self.config_manager.load_config(config_path)
                    self._logger.info("应用初始化完成（使用自定义配置）")
                else:
                    self._logger.info("应用初始化完成（使用默认配置）")
                
            except Exception as e:
                self._logger.error(f"应用初始化失败: {e}")
                raise
        
        def generate_single_hologram(self, params: ModeParameters) -> Dict[str, Any]:
            """生成单个全息图"""
            try:
                self._logger.info(f"开始生成单个全息图: l={params.l}, p={params.p}")
                
                # 验证参数
                errors = params.validate()
                if errors:
                    raise ValidationError(f"参数验证失败: {'; '.join(errors)}")
                
                # 获取配置
                system_config = self.config_manager.get_system_config()
                optical_config = self.config_manager.get_optical_config()
                
                # 生成结果
                phase_map = self.algorithm_engine.generate_phase_map(
                    params, optical_config, system_config
                )
                hologram = self.algorithm_engine.generate_hologram(
                    params, optical_config, system_config
                )
                
                result = {
                    'parameters': params,
                    'phase_map': phase_map,
                    'hologram': hologram,
                    'success': True
                }
                
                self._logger.info("单个全息图生成完成")
                return result
                
            except Exception as e:
                self._logger.error(f"单个全息图生成失败: {e}")
                raise
        
        def generate_batch_holograms(self, parameters: List[ModeParameters]) -> List[Dict[str, Any]]:
            """生成批量全息图"""
            try:
                self._logger.info(f"开始批量生成: {len(parameters)} 组参数")
                
                results = self.data_processor.process_batch(parameters)
                
                successful_count = sum(1 for r in results if r.get('success', False))
                self._logger.info(f"批量生成完成: {successful_count}/{len(results)} 成功")
                
                return results
                
            except Exception as e:
                self._logger.error(f"批量生成失败: {e}")
                raise
        
        def load_parameters_from_file(self, file_path: Path) -> List[ModeParameters]:
            """从文件加载参数"""
            try:
                if file_path.suffix.lower() == '.json':
                    return self._load_parameters_from_json(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    return self._load_parameters_from_excel(file_path)
                else:
                    raise ValueError(f"不支持的文件格式: {file_path.suffix}")
                
            except Exception as e:
                self._logger.error(f"加载参数失败: {e}")
                raise
        
        def _load_parameters_from_json(self, file_path: Path) -> List[ModeParameters]:
            """从JSON文件加载参数"""
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            parameters = []
            for item in data.get('parameters', []):
                param = ModeParameters(**item)
                parameters.append(param)
            
            return parameters
        
        def _load_parameters_from_excel(self, file_path: Path) -> List[ModeParameters]:
            """从Excel文件加载参数"""
            # 这里会调用Excel处理模块
            # 为了演示，返回示例数据
            return [
                ModeParameters(l=1, p=0, amplitude=1.0, phase=0.0),
                ModeParameters(l=-1, p=0, amplitude=1.0, phase=0.5),
            ]
        
        def save_configuration(self, config_path: Path):
            """保存当前配置"""
            try:
                config_data = {
                    'timestamp': time.time(),
                    'version': '1.0.0'
                }
                
                self.config_manager.save_config(config_data, config_path)
                self._logger.info(f"配置已保存: {config_path}")
                
            except Exception as e:
                self._logger.error(f"保存配置失败: {e}")
                raise
        
        def get_system_status(self) -> Dict[str, Any]:
            """获取系统状态"""
            system_config = self.config_manager.get_system_config()
            optical_config = self.config_manager.get_optical_config()
            
            return {
                'system_config': asdict(system_config),
                'optical_config': asdict(optical_config),
                'status': 'ready',
                'timestamp': time.time()
            }
    
    # ==================== 表示层接口 (Presentation Layer Interface) ====================
    
    @runtime_checkable
    class UIController(Protocol):
        """UI控制器接口"""
        def initialize_ui(self) -> None:
            """初始化UI"""
            ...
        
        def update_progress(self, current: int, total: int, message: str) -> None:
            """更新进度"""
            ...
        
        def show_results(self, results: List[Any]) -> None:
            """显示结果"""
            ...
        
        def show_error(self, error_message: str) -> None:
            """显示错误"""
            ...
    
    return (OAMApplicationService, ConfigManager, AlgorithmEngine, 
            DataProcessor, UIController, SystemConfig, OpticalConfig, ModeParameters)

# 使用示例
def demonstrate_architecture():
    """演示架构使用"""
    import time
    
    # 获取架构组件
    (OAMApplicationService, ConfigManager, AlgorithmEngine, 
     DataProcessor, UIController, SystemConfig, OpticalConfig, ModeParameters) = project_architecture_design()
    
    print("=== OAM项目架构演示 ===")
    
    # 创建应用服务
    app_service = OAMApplicationService()
    
    # 初始化应用
    app_service.initialize_application()
    
    # 获取系统状态
    status = app_service.get_system_status()
    print(f"系统状态: {status['status']}")
    print(f"图像尺寸: {status['system_config']['image_width']}x{status['system_config']['image_height']}")
    
    # 生成单个全息图
    single_param = ModeParameters(l=1, p=0, amplitude=1.0, phase=0.0)
    result = app_service.generate_single_hologram(single_param)
    print(f"单个全息图生成: {'成功' if result['success'] else '失败'}")
    
    # 批量生成
    batch_params = [
        ModeParameters(l=1, p=0, amplitude=1.0, phase=0.0),
        ModeParameters(l=-1, p=0, amplitude=1.0, phase=0.5),
        ModeParameters(l=2, p=1, amplitude=0.707, phase=0.25),
    ]
    
    batch_results = app_service.generate_batch_holograms(batch_params)
    successful_count = sum(1 for r in batch_results if r.get('success', False))
    print(f"批量生成: {successful_count}/{len(batch_results)} 成功")
    
    print("架构演示完成")
    
    return app_service
```

### 13.2 依赖管理与模块解耦

#### 13.2.1 依赖注入容器实现
```python
from typing import TypeVar, Type, Dict, Any, Callable, Optional
import inspect
from functools import wraps

def dependency_injection_system():
    """依赖注入系统实现"""
    
    T = TypeVar('T')
    
    class DIContainer:
        """依赖注入容器"""
        
        def __init__(self):
            self._services: Dict[Type, Any] = {}
            self._factories: Dict[Type, Callable] = {}
            self._singletons: Dict[Type, Any] = {}
            self._scoped: Dict[Type, Any] = {}
            
        def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
            """注册单例服务"""
            self._services[interface] = implementation
            self._factories[interface] = 'singleton'
            
        def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
            """注册瞬态服务"""
            self._services[interface] = implementation
            self._factories[interface] = 'transient'
            
        def register_scoped(self, interface: Type[T], implementation: Type[T]) -> None:
            """注册作用域服务"""
            self._services[interface] = implementation
            self._factories[interface] = 'scoped'
            
        def register_instance(self, interface: Type[T], instance: T) -> None:
            """注册实例"""
            self._singletons[interface] = instance
            
        def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
            """注册工厂方法"""
            self._factories[interface] = factory
            
        def resolve(self, interface: Type[T]) -> T:
            """解析服务"""
            # 检查是否有已注册的实例
            if interface in self._singletons:
                return self._singletons[interface]
            
            # 检查是否有注册的服务
            if interface not in self._services and interface not in self._factories:
                raise ValueError(f"服务未注册: {interface}")
            
            # 获取工厂类型
            factory_type = self._factories.get(interface)
            
            if factory_type == 'singleton':
                if interface not in self._singletons:
                    self._singletons[interface] = self._create_instance(interface)
                return self._singletons[interface]
            
            elif factory_type == 'scoped':
                if interface not in self._scoped:
                    self._scoped[interface] = self._create_instance(interface)
                return self._scoped[interface]
            
            elif factory_type == 'transient':
                return self._create_instance(interface)
            
            elif callable(factory_type):
                return factory_type()
            
            else:
                return self._create_instance(interface)
        
        def _create_instance(self, interface: Type[T]) -> T:
            """创建实例"""
            implementation = self._services[interface]
            
            # 获取构造函数参数
            sig = inspect.signature(implementation.__init__)
            parameters = sig.parameters
            
            # 解析构造函数依赖
            kwargs = {}
            for param_name, param in parameters.items():
                if param_name == 'self':
                    continue
                
                param_type = param.annotation
                if param_type != inspect.Parameter.empty:
                    kwargs[param_name] = self.resolve(param_type)
            
            return implementation(**kwargs)
        
        def clear_scoped(self):
            """清除作用域服务"""
            self._scoped.clear()
    
    # 装饰器支持
    def inject(*dependencies):
        """依赖注入装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 从容器解析依赖
                container = getattr(wrapper, '_container', None)
                if container:
                    for i, dep_type in enumerate(dependencies):
                        if i < len(args):
                            continue  # 已提供的参数跳过
                        kwargs[f'dep_{i}'] = container.resolve(dep_type)
                
                return func(*args, **kwargs)
            
            wrapper._dependencies = dependencies
            return wrapper
        return decorator
    
    # 配置类
    class DIConfiguration:
        """依赖注入配置"""
        
        @staticmethod
        def configure_oam_services(container: DIContainer):
            """配置OAM项目的服务"""
            from project_architecture_design import (
                JSONConfigManager, LGAlgorithmEngine, BatchDataProcessor,
                ConfigManager, AlgorithmEngine, DataProcessor
            )
            
            # 注册服务
            container.register_singleton(ConfigManager, JSONConfigManager)
            container.register_singleton(AlgorithmEngine, LGAlgorithmEngine)
            container.register_transient(DataProcessor, BatchDataProcessor)
    
    return DIContainer, inject, DIConfiguration

def service_locator_pattern():
    """服务定位器模式"""
    
    class ServiceLocator:
        """服务定位器"""
        
        _instance = None
        _services: Dict[Type, Any] = {}
        
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
        
        @classmethod
        def register(cls, interface: Type[T], implementation: T):
            """注册服务"""
            cls._services[interface] = implementation
        
        @classmethod
        def get(cls, interface: Type[T]) -> T:
            """获取服务"""
            if interface not in cls._services:
                raise ValueError(f"服务未注册: {interface}")
            return cls._services[interface]
        
        @classmethod
        def clear(cls):
            """清除所有服务"""
            cls._services.clear()
    
    return ServiceLocator

def module_loading_system():
    """模块动态加载系统"""
    
    class ModuleLoader:
        """模块加载器"""
        
        def __init__(self):
            self._loaded_modules = {}
            self._module_configs = {}
        
        def register_module(self, module_name: str, module_config: Dict[str, Any]):
            """注册模块配置"""
            self._module_configs[module_name] = module_config
        
        def load_module(self, module_name: str) -> Any:
            """动态加载模块"""
            if module_name in self._loaded_modules:
                return self._loaded_modules[module_name]
            
            if module_name not in self._module_configs:
                raise ValueError(f"模块配置未找到: {module_name}")
            
            config = self._module_configs[module_name]
            
            try:
                # 动态导入
                import importlib
                module_path = config.get('module_path')
                class_name = config.get('class_name')
                
                module = importlib.import_module(module_path)
                module_class = getattr(module, class_name)
                
                # 创建实例
                init_params = config.get('init_params', {})
                instance = module_class(**init_params)
                
                self._loaded_modules[module_name] = instance
                return instance
                
            except Exception as e:
                raise ImportError(f"模块加载失败 {module_name}: {e}")
        
        def unload_module(self, module_name: str):
            """卸载模块"""
            if module_name in self._loaded_modules:
                del self._loaded_modules[module_name]
        
        def reload_module(self, module_name: str) -> Any:
            """重新加载模块"""
            self.unload_module(module_name)
            return self.load_module(module_name)
        
        def list_loaded_modules(self) -> List[str]:
            """列出已加载的模块"""
            return list(self._loaded_modules.keys())
    
    return ModuleLoader

# 使用示例
def demonstrate_dependency_injection():
    """演示依赖注入"""
    print("=== 依赖注入系统演示 ===")
    
    # 获取依赖注入组件
    DIContainer, inject, DIConfiguration = dependency_injection_system()
    
    # 创建容器
    container = DIContainer()
    
    # 配置服务（模拟）
    class ILogger:
        def log(self, message: str): pass
    
    class ConsoleLogger:
        def log(self, message: str):
            print(f"[LOG] {message}")
    
    class IRepository:
        def save(self, data: Any): pass
    
    class FileRepository:
        def __init__(self, logger: ILogger):
            self.logger = logger
        
        def save(self, data: Any):
            self.logger.log(f"保存数据: {data}")
    
    class BusinessService:
        def __init__(self, repository: IRepository, logger: ILogger):
            self.repository = repository
            self.logger = logger
        
        def process(self, data: Any):
            self.logger.log("开始处理业务")
            self.repository.save(data)
            self.logger.log("处理完成")
    
    # 注册服务
    container.register_singleton(ILogger, ConsoleLogger)
    container.register_transient(IRepository, FileRepository)
    container.register_transient(BusinessService, BusinessService)
    
    # 解析并使用服务
    service = container.resolve(BusinessService)
    service.process("测试数据")
    
    print("依赖注入演示完成")
    
    return container
```

---

## 第14章：错误处理与调试技巧

### 14.1 异常处理架构设计

#### 14.1.1 分层异常处理体系
```python
import traceback
import sys
from typing import Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime

def comprehensive_error_handling():
    """全面的错误处理系统"""
    
    # ==================== 异常层次结构 ====================
    
    class OAMBaseException(Exception):
        """OAM项目基础异常"""
        
        def __init__(self, message: str, error_code: str = None, 
                     details: Dict[str, Any] = None, cause: Exception = None):
            super().__init__(message)
            self.message = message
            self.error_code = error_code or self.__class__.__name__
            self.details = details or {}
            self.cause = cause
            self.timestamp = datetime.now()
        
        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            return {
                'error_type': self.__class__.__name__,
                'error_code': self.error_code,
                'message': self.message,
                'details': self.details,
                'timestamp': self.timestamp.isoformat(),
                'cause': str(self.cause) if self.cause else None
            }
        
        def __str__(self) -> str:
            return f"[{self.error_code}] {self.message}"
    
    # 配置相关异常
    class ConfigurationException(OAMBaseException):
        """配置异常"""
        pass
    
    class InvalidConfigValueException(ConfigurationException):
        """无效配置值异常"""
        pass
    
    class MissingConfigException(ConfigurationException):
        """缺少配置异常"""
        pass
    
    # 验证相关异常
    class ValidationException(OAMBaseException):
        """验证异常"""
        pass
    
    class ParameterValidationException(ValidationException):
        """参数验证异常"""
        pass
    
    class FileValidationException(ValidationException):
        """文件验证异常"""
        pass
    
    # 处理相关异常
    class ProcessingException(OAMBaseException):
        """处理异常"""
        pass
    
    class AlgorithmException(ProcessingException):
        """算法异常"""
        pass
    
    class ResourceException(ProcessingException):
        """资源异常"""
        pass
    
    class InsufficientMemoryException(ResourceException):
        """内存不足异常"""
        pass
    
    class DiskSpaceException(ResourceException):
        """磁盘空间异常"""
        pass
    
    # 文件操作异常
    class FileOperationException(OAMBaseException):
        """文件操作异常"""
        pass
    
    class FileNotFoundOAMException(FileOperationException):
        """文件未找到异常"""
        pass
    
    class FileAccessException(FileOperationException):
        """文件访问异常"""
        pass
    
    class FileFormatException(FileOperationException):
        """文件格式异常"""
        pass
    
    # UI相关异常
    class UIException(OAMBaseException):
        """UI异常"""
        pass
    
    class UserCancelledException(UIException):
        """用户取消异常"""
        pass
    
    # ==================== 错误级别定义 ====================
    
    class ErrorSeverity(Enum):
        """错误严重程度"""
        LOW = "low"           # 轻微错误，不影响主要功能
        MEDIUM = "medium"     # 中等错误，影响部分功能
        HIGH = "high"         # 严重错误，影响核心功能
        CRITICAL = "critical" # 致命错误，系统无法继续运行
    
    @dataclass
    class ErrorContext:
        """错误上下文"""
        function_name: str
        module_name: str
        line_number: int
        local_variables: Dict[str, Any]
        stack_trace: str
        severity: ErrorSeverity
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'function_name': self.function_name,
                'module_name': self.module_name,
                'line_number': self.line_number,
                'local_variables': self.local_variables,
                'stack_trace': self.stack_trace,
                'severity': self.severity.value
            }
    
    # ==================== 错误处理器 ====================
    
    class ErrorHandler(ABC):
        """错误处理器抽象基类"""
        
        @abstractmethod
        def handle_error(self, exception: Exception, context: ErrorContext) -> bool:
            """
            处理错误
            
            Returns:
                bool: True表示错误已处理，False表示需要继续传播
            """
            pass
        
        @abstractmethod
        def can_handle(self, exception: Exception) -> bool:
            """判断是否能处理此异常"""
            pass
    
    class LoggingErrorHandler(ErrorHandler):
        """日志错误处理器"""
        
        def __init__(self, logger: logging.Logger):
            self.logger = logger
        
        def can_handle(self, exception: Exception) -> bool:
            """所有异常都记录日志"""
            return True
        
        def handle_error(self, exception: Exception, context: ErrorContext) -> bool:
            """记录错误日志"""
            error_info = {
                'exception_type': exception.__class__.__name__,
                'message': str(exception),
                'context': context.to_dict()
            }
            
            if context.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"致命错误: {error_info}")
            elif context.severity == ErrorSeverity.HIGH:
                self.logger.error(f"严重错误: {error_info}")
            elif context.severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"警告: {error_info}")
            else:
                self.logger.info(f"信息: {error_info}")
            
            return False  # 继续传播
    
    class FileErrorHandler(ErrorHandler):
        """文件错误处理器"""
        
        def __init__(self, error_log_path: Path):
            self.error_log_path = error_log_path
            self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        def can_handle(self, exception: Exception) -> bool:
            """处理所有严重级别以上的错误"""
            return hasattr(exception, 'severity') and exception.severity in [
                ErrorSeverity.HIGH, ErrorSeverity.CRITICAL
            ]
        
        def handle_error(self, exception: Exception, context: ErrorContext) -> bool:
            """将错误写入文件"""
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'exception': {
                    'type': exception.__class__.__name__,
                    'message': str(exception),
                    'details': getattr(exception, 'details', {})
                },
                'context': context.to_dict()
            }
            
            try:
                with open(self.error_log_path, 'a', encoding='utf-8') as f:
                    json.dump(error_record, f, ensure_ascii=False)
                    f.write('\n')
            except Exception as e:
                # 如果无法写入错误日志，输出到stderr
                print(f"无法写入错误日志: {e}", file=sys.stderr)
            
            return False  # 继续传播
    
    class NotificationErrorHandler(ErrorHandler):
        """通知错误处理器"""
        
        def __init__(self, notification_callback: Callable[[str, ErrorSeverity], None]):
            self.notification_callback = notification_callback
        
        def can_handle(self, exception: Exception) -> bool:
            """处理高级别错误"""
            return hasattr(exception, 'severity') and exception.severity in [
                ErrorSeverity.HIGH, ErrorSeverity.CRITICAL
            ]
        
        def handle_error(self, exception: Exception, context: ErrorContext) -> bool:
            """发送通知"""
            message = f"错误发生: {exception.message if hasattr(exception, 'message') else str(exception)}"
            severity = getattr(exception, 'severity', ErrorSeverity.MEDIUM)
            
            try:
                self.notification_callback(message, severity)
            except Exception as e:
                print(f"发送通知失败: {e}", file=sys.stderr)
            
            return False  # 继续传播
    
    class RecoveryErrorHandler(ErrorHandler):
        """恢复错误处理器"""
        
        def __init__(self):
            self.recovery_strategies = {}
        
        def register_recovery_strategy(self, exception_type: type, 
                                     strategy: Callable[[Exception], Any]):
            """注册恢复策略"""
            self.recovery_strategies[exception_type] = strategy
        
        def can_handle(self, exception: Exception) -> bool:
            """检查是否有对应的恢复策略"""
            return type(exception) in self.recovery_strategies
        
        def handle_error(self, exception: Exception, context: ErrorContext) -> bool:
            """尝试恢复"""
            strategy = self.recovery_strategies.get(type(exception))
            if strategy:
                try:
                    strategy(exception)
                    return True  # 已恢复，停止传播
                except Exception as recovery_error:
                    print(f"恢复策略失败: {recovery_error}", file=sys.stderr)
            
            return False  # 继续传播
    
    # ==================== 错误管理器 ====================
    
    class ErrorManager:
        """错误管理器"""
        
        def __init__(self):
            self.handlers: List[ErrorHandler] = []
            self.error_history: List[Dict[str, Any]] = []
            self.max_history_size = 1000
        
        def add_handler(self, handler: ErrorHandler):
            """添加错误处理器"""
            self.handlers.append(handler)
        
        def handle_exception(self, exception: Exception, 
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM):
            """处理异常"""
            # 创建错误上下文
            context = self._create_error_context(exception, severity)
            
            # 记录到历史
            self._add_to_history(exception, context)
            
            # 依次调用处理器
            handled = False
            for handler in self.handlers:
                if handler.can_handle(exception):
                    try:
                        if handler.handle_error(exception, context):
                            handled = True
                            break
                    except Exception as handler_error:
                        print(f"错误处理器异常: {handler_error}", file=sys.stderr)
            
            return handled
        
        def _create_error_context(self, exception: Exception, 
                                severity: ErrorSeverity) -> ErrorContext:
            """创建错误上下文"""
            frame = sys._getframe(2)  # 调用堆栈向上2层
            
            # 获取局部变量（安全地）
            local_vars = {}
            try:
                for key, value in frame.f_locals.items():
                    try:
                        # 尝试序列化值
                        json.dumps(value, default=str)
                        local_vars[key] = str(value)[:200]  # 限制长度
                    except:
                        local_vars[key] = f"<{type(value).__name__}>"
            except:
                local_vars = {"error": "无法获取局部变量"}
            
            return ErrorContext(
                function_name=frame.f_code.co_name,
                module_name=frame.f_code.co_filename,
                line_number=frame.f_lineno,
                local_variables=local_vars,
                stack_trace=traceback.format_exc(),
                severity=severity
            )
        
        def _add_to_history(self, exception: Exception, context: ErrorContext):
            """添加到错误历史"""
            record = {
                'timestamp': datetime.now().isoformat(),
                'exception_type': exception.__class__.__name__,
                'message': str(exception),
                'context': context.to_dict()
            }
            
            self.error_history.append(record)
            
            # 限制历史大小
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
        
        def get_error_statistics(self) -> Dict[str, Any]:
            """获取错误统计"""
            if not self.error_history:
                return {"total_errors": 0}
            
            # 按类型统计
            type_counts = {}
            severity_counts = {}
            
            for record in self.error_history:
                exc_type = record['exception_type']
                type_counts[exc_type] = type_counts.get(exc_type, 0) + 1
                
                severity = record['context']['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                'total_errors': len(self.error_history),
                'by_type': type_counts,
                'by_severity': severity_counts,
                'latest_error': self.error_history[-1] if self.error_history else None
            }
        
        def export_error_report(self, file_path: Path):
            """导出错误报告"""
            report = {
                'generated_at': datetime.now().isoformat(),
                'statistics': self.get_error_statistics(),
                'error_history': self.error_history
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
    
    # ==================== 装饰器支持 ====================
    
    def error_boundary(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      reraise: bool = True,
                      recovery_value: Any = None):
        """错误边界装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 获取全局错误管理器
                    error_manager = getattr(wrapper, '_error_manager', None)
                    if error_manager:
                        handled = error_manager.handle_exception(e, severity)
                        
                        if handled and not reraise:
                            return recovery_value
                    
                    if reraise:
                        raise
                    else:
                        return recovery_value
            
            return wrapper
        return decorator
    
    def safe_execute(func: Callable, *args, default_value: Any = None, 
                    error_manager: ErrorManager = None, **kwargs) -> Any:
        """安全执行函数"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if error_manager:
                error_manager.handle_exception(e)
            return default_value
    
    return (ErrorManager, OAMBaseException, ErrorSeverity, ErrorContext,
            LoggingErrorHandler, FileErrorHandler, NotificationErrorHandler,
            RecoveryErrorHandler, error_boundary, safe_execute)

# 使用示例
def demonstrate_error_handling():
    """演示错误处理系统"""
    print("=== 错误处理系统演示 ===")
    
    # 获取错误处理组件
    (ErrorManager, OAMBaseException, ErrorSeverity, ErrorContext,
     LoggingErrorHandler, FileErrorHandler, NotificationErrorHandler,
     RecoveryErrorHandler, error_boundary, safe_execute) = comprehensive_error_handling()
    
    # 创建错误管理器
    error_manager = ErrorManager()
    
    # 设置日志
    import logging
    logger = logging.getLogger("OAM_Demo")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 添加错误处理器
    error_manager.add_handler(LoggingErrorHandler(logger))
    error_manager.add_handler(FileErrorHandler(Path("error_log.json")))
    
    def notification_callback(message: str, severity: ErrorSeverity):
        print(f"[通知] {severity.value.upper()}: {message}")
    
    error_manager.add_handler(NotificationErrorHandler(notification_callback))
    
    # 注册恢复策略
    recovery_handler = RecoveryErrorHandler()
    
    def recover_from_file_not_found(exception):
        print("恢复策略: 创建默认文件")
        # 这里实现具体的恢复逻辑
    
    recovery_handler.register_recovery_strategy(FileNotFoundError, recover_from_file_not_found)
    error_manager.add_handler(recovery_handler)
    
    # 测试错误处理
    class ParameterValidationException(OAMBaseException):
        pass
    
    @error_boundary(severity=ErrorSeverity.MEDIUM, reraise=False, recovery_value="默认值")
    def test_function(value: int):
        if value < 0:
            raise ParameterValidationException(
                "参数值不能为负数",
                error_code="INVALID_PARAM",
                details={"parameter": "value", "received": value}
            )
        return f"处理结果: {value}"
    
    # 绑定错误管理器到装饰器
    test_function._error_manager = error_manager
    
    # 测试正常情况
    result1 = test_function(10)
    print(f"正常结果: {result1}")
    
    # 测试异常情况
    result2 = test_function(-5)
    print(f"异常恢复结果: {result2}")
    
    # 显示错误统计
    stats = error_manager.get_error_statistics()
    print(f"错误统计: {stats}")
    
    # 导出错误报告
    error_manager.export_error_report(Path("error_report.json"))
    print("错误报告已导出")
    
    return error_manager
```

### 14.2 调试工具与技巧

#### 14.2.1 高级调试技术
```python
import pdb
import cProfile
import traceback
import sys
import time
import functools
from typing import Any, Dict, List, Callable, Optional
from contextlib import contextmanager
import logging
import threading
from dataclasses import dataclass

def advanced_debugging_tools():
    """高级调试工具"""
    
    # ==================== 性能分析工具 ====================
    
    class PerformanceProfiler:
        """性能分析器"""
        
        def __init__(self):
            self.profiles = {}
            self.call_counts = {}
            self.execution_times = {}
        
        def profile_function(self, func_name: str = None):
            """函数性能分析装饰器"""
            def decorator(func):
                name = func_name or f"{func.__module__}.{func.__name__}"
                
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        end_time = time.perf_counter()
                        execution_time = end_time - start_time
                        
                        # 记录统计信息
                        if name not in self.call_counts:
                            self.call_counts[name] = 0
                            self.execution_times[name] = []
                        
                        self.call_counts[name] += 1
                        self.execution_times[name].append(execution_time)
                
                return wrapper
            return decorator
        
        def profile_context(self, name: str):
            """上下文管理器形式的性能分析"""
            return self._ProfileContext(self, name)
        
        class _ProfileContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                execution_time = end_time - self.start_time
                
                if self.name not in self.profiler.call_counts:
                    self.profiler.call_counts[self.name] = 0
                    self.profiler.execution_times[self.name] = []
                
                self.profiler.call_counts[self.name] += 1
                self.profiler.execution_times[self.name].append(execution_time)
        
        def get_statistics(self) -> Dict[str, Dict[str, Any]]:
            """获取性能统计"""
            stats = {}
            
            for name in self.call_counts:
                times = self.execution_times[name]
                stats[name] = {
                    'call_count': self.call_counts[name],
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
            
            return stats
        
        def print_report(self):
            """打印性能报告"""
            stats = self.get_statistics()
            
            print("=== 性能分析报告 ===")
            print(f"{'函数名':<40} {'调用次数':<10} {'总时间(s)':<12} {'平均时间(s)':<12} {'最小时间(s)':<12} {'最大时间(s)':<12}")
            print("-" * 120)
            
            # 按总时间排序
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
            
            for name, stat in sorted_stats:
                print(f"{name:<40} {stat['call_count']:<10} {stat['total_time']:<12.6f} "
                      f"{stat['average_time']:<12.6f} {stat['min_time']:<12.6f} {stat['max_time']:<12.6f}")
    
    # ==================== 内存监控工具 ====================
    
    class MemoryMonitor:
        """内存监控器"""
        
        def __init__(self):
            self.snapshots = []
            self.peak_memory = 0
        
        def take_snapshot(self, label: str = None):
            """获取内存快照"""
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                
                snapshot = {
                    'label': label or f"snapshot_{len(self.snapshots)}",
                    'timestamp': time.time(),
                    'rss': memory_info.rss,  # 物理内存
                    'vms': memory_info.vms,  # 虚拟内存
                    'percent': process.memory_percent()
                }
                
                self.snapshots.append(snapshot)
                
                if memory_info.rss > self.peak_memory:
                    self.peak_memory = memory_info.rss
                
                return snapshot
                
            except ImportError:
                print("警告: 需要安装psutil库进行内存监控")
                return None
        
        def monitor_function(self, func_name: str = None):
            """函数内存监控装饰器"""
            def decorator(func):
                name = func_name or f"{func.__module__}.{func.__name__}"
                
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    self.take_snapshot(f"{name}_start")
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        self.take_snapshot(f"{name}_end")
                
                return wrapper
            return decorator
        
        def get_memory_usage_report(self) -> Dict[str, Any]:
            """获取内存使用报告"""
            if not self.snapshots:
                return {"error": "没有内存快照"}
            
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]
            
            return {
                'snapshots_count': len(self.snapshots),
                'peak_memory_mb': self.peak_memory / 1024 / 1024,
                'initial_memory_mb': first_snapshot['rss'] / 1024 / 1024,
                'final_memory_mb': last_snapshot['rss'] / 1024 / 1024,
                'memory_delta_mb': (last_snapshot['rss'] - first_snapshot['rss']) / 1024 / 1024,
                'snapshots': self.snapshots
            }
    
    # ==================== 调用栈跟踪器 ====================
    
    class CallStackTracer:
        """调用栈跟踪器"""
        
        def __init__(self):
            self.trace_active = False
            self.call_stack = []
            self.max_depth = 50
        
        def start_trace(self):
            """开始跟踪"""
            self.trace_active = True
            self.call_stack = []
            sys.settrace(self._trace_calls)
        
        def stop_trace(self):
            """停止跟踪"""
            self.trace_active = False
            sys.settrace(None)
        
        def _trace_calls(self, frame, event, arg):
            """跟踪回调函数"""
            if not self.trace_active or len(self.call_stack) > self.max_depth:
                return
            
            if event == 'call':
                func_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                line_number = frame.f_lineno
                
                call_info = {
                    'event': 'call',
                    'function': func_name,
                    'file': filename,
                    'line': line_number,
                    'timestamp': time.time(),
                    'locals': dict(frame.f_locals)
                }
                
                self.call_stack.append(call_info)
                
            elif event == 'return':
                if self.call_stack:
                    call_info = self.call_stack[-1]
                    call_info['return_value'] = arg
                    call_info['duration'] = time.time() - call_info['timestamp']
            
            return self._trace_calls
        
        def get_call_tree(self) -> List[Dict[str, Any]]:
            """获取调用树"""
            return self.call_stack
    
    # ==================== 变量监视器 ====================
    
    class VariableWatcher:
        """变量监视器"""
        
        def __init__(self):
            self.watched_variables = {}
            self.change_history = []
        
        def watch_variable(self, var_name: str, var_ref: Any):
            """监视变量"""
            self.watched_variables[var_name] = {
                'reference': var_ref,
                'last_value': getattr(var_ref, var_name, None),
                'change_count': 0
            }
        
        def check_changes(self):
            """检查变量变化"""
            changes = []
            
            for var_name, info in self.watched_variables.items():
                try:
                    current_value = getattr(info['reference'], var_name, None)
                    last_value = info['last_value']
                    
                    if current_value != last_value:
                        change_record = {
                            'variable': var_name,
                            'old_value': last_value,
                            'new_value': current_value,
                            'timestamp': time.time(),
                            'change_number': info['change_count'] + 1
                        }
                        
                        changes.append(change_record)
                        self.change_history.append(change_record)
                        
                        # 更新记录
                        info['last_value'] = current_value
                        info['change_count'] += 1
                
                except Exception as e:
                    print(f"监视变量 {var_name} 时出错: {e}")
            
            return changes
    
    # ==================== 断点管理器 ====================
    
    class BreakpointManager:
        """断点管理器"""
        
        def __init__(self):
            self.breakpoints = {}
            self.conditions = {}
            self.hit_counts = {}
        
        def set_breakpoint(self, func: Callable, condition: Callable = None):
            """设置断点"""
            func_name = f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 检查条件
                if condition is None or condition(*args, **kwargs):
                    # 记录命中
                    self.hit_counts[func_name] = self.hit_counts.get(func_name, 0) + 1
                    
                    print(f"断点命中: {func_name} (第{self.hit_counts[func_name]}次)")
                    print(f"参数: args={args}, kwargs={kwargs}")
                    
                    # 进入调试器
                    pdb.set_trace()
                
                return func(*args, **kwargs)
            
            self.breakpoints[func_name] = wrapper
            return wrapper
        
        def conditional_breakpoint(self, condition: Callable):
            """条件断点装饰器"""
            def decorator(func):
                return self.set_breakpoint(func, condition)
            return decorator
    
    # ==================== 调试会话管理器 ====================
    
    class DebugSession:
        """调试会话管理器"""
        
        def __init__(self):
            self.profiler = PerformanceProfiler()
            self.memory_monitor = MemoryMonitor()
            self.call_tracer = CallStackTracer()
            self.variable_watcher = VariableWatcher()
            self.breakpoint_manager = BreakpointManager()
            self.session_active = False
        
        def start_session(self, enable_tracing: bool = False, 
                         enable_profiling: bool = True,
                         enable_memory_monitoring: bool = True):
            """开始调试会话"""
            self.session_active = True
            
            if enable_tracing:
                self.call_tracer.start_trace()
            
            if enable_memory_monitoring:
                self.memory_monitor.take_snapshot("session_start")
            
            print(f"调试会话已开始 (跟踪: {enable_tracing}, 性能分析: {enable_profiling}, 内存监控: {enable_memory_monitoring})")
        
        def end_session(self):
            """结束调试会话"""
            if not self.session_active:
                return
            
            self.session_active = False
            self.call_tracer.stop_trace()
            self.memory_monitor.take_snapshot("session_end")
            
            # 生成报告
            print("\n=== 调试会话报告 ===")
            
            # 性能报告
            self.profiler.print_report()
            
            # 内存报告
            memory_report = self.memory_monitor.get_memory_usage_report()
            print(f"\n内存使用: 初始 {memory_report.get('initial_memory_mb', 0):.2f}MB, "
                  f"最终 {memory_report.get('final_memory_mb', 0):.2f}MB, "
                  f"峰值 {memory_report.get('peak_memory_mb', 0):.2f}MB")
            
            # 变量变化
            changes = self.variable_watcher.check_changes()
            if changes:
                print(f"\n检测到 {len(changes)} 个变量变化")
        
        @contextmanager
        def debug_context(self, name: str):
            """调试上下文管理器"""
            print(f"进入调试上下文: {name}")
            
            with self.profiler.profile_context(name):
                self.memory_monitor.take_snapshot(f"{name}_start")
                
                try:
                    yield self
                finally:
                    self.memory_monitor.take_snapshot(f"{name}_end")
                    print(f"离开调试上下文: {name}")
    
    return (PerformanceProfiler, MemoryMonitor, CallStackTracer, 
            VariableWatcher, BreakpointManager, DebugSession)

# 使用示例
def demonstrate_debugging_tools():
    """演示调试工具"""
    print("=== 调试工具演示 ===")
    
    # 获取调试工具
    (PerformanceProfiler, MemoryMonitor, CallStackTracer, 
     VariableWatcher, BreakpointManager, DebugSession) = advanced_debugging_tools()
    
    # 创建调试会话
    debug_session = DebugSession()
    debug_session.start_session(enable_tracing=False)  # 关闭跟踪以避免过多输出
    
    # 测试性能分析
    @debug_session.profiler.profile_function("test_calculation")
    def slow_calculation(n: int) -> int:
        """模拟慢速计算"""
        result = 0
        for i in range(n):
            result += i * i
        return result
    
    # 测试内存监控
    @debug_session.memory_monitor.monitor_function("memory_test")
    def memory_intensive_function():
        """模拟内存密集型函数"""
        # 创建大列表
        big_list = [i for i in range(100000)]
        return len(big_list)
    
    # 执行测试
    with debug_session.debug_context("calculation_test"):
        result1 = slow_calculation(10000)
        result2 = slow_calculation(20000)
        result3 = memory_intensive_function()
    
    # 结束会话并查看报告
    debug_session.end_session()
    
    print(f"计算结果: {result1}, {result2}, {result3}")
    
    return debug_session
```

---

**文档状态**: 第13-14章详细内容已完成 ✅

**已完成内容**:
- 完整的项目架构设计
- 依赖注入系统实现
- 分层异常处理体系
- 高级调试工具集

**下一步**: 完成第15章性能优化与测试策略







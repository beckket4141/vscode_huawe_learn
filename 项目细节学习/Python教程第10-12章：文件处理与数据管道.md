# Python 教程第10-12章：文件处理与数据管道

> **本文档涵盖**：第10章 文件IO与路径管理、第11章 Excel处理与批量数据管道、第12章 配置管理与序列化

---

## 第10章：文件 IO 与路径管理

### 10.1 pathlib 模块的高级应用

#### 10.1.1 路径操作与OAM项目文件管理
```python
from pathlib import Path, PurePath
import os
import shutil
import tempfile
import time
from typing import List, Dict, Optional, Iterator, Union
import json
import csv

def comprehensive_pathlib_usage():
    """pathlib模块全面应用教学"""
    
    class ProjectFileManager:
        """项目文件管理器 - 展示pathlib在OAM项目中的应用"""
        
        def __init__(self, project_root: Union[str, Path] = None):
            """
            初始化文件管理器
            
            Args:
                project_root: 项目根目录路径
            """
            self.project_root = Path(project_root) if project_root else Path.cwd()
            self.ensure_project_structure()
            
            # 定义项目目录结构
            self.directories = {
                'output': self.project_root / 'LG_output',
                'config': self.project_root / 'config',
                'data': self.project_root / 'data',
                'logs': self.project_root / 'logs',
                'temp': self.project_root / 'temp',
                'backup': self.project_root / 'backup',
                'cache': self.project_root / 'cache'
            }
            
            # 支持的文件类型
            self.supported_formats = {
                'image': ['.bmp', '.png', '.tiff', '.jpg', '.jpeg'],
                'config': ['.json', '.yaml', '.yml', '.ini'],
                'data': ['.csv', '.xlsx', '.xls', '.txt', '.dat'],
                'log': ['.log', '.txt'],
                'backup': ['.zip', '.tar', '.gz']
            }
        
        def ensure_project_structure(self):
            """确保项目目录结构存在"""
            required_dirs = [
                'LG_output', 'config', 'data', 'logs', 
                'temp', 'backup', 'cache'
            ]
            
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True, parents=True)
                
                # 创建.gitkeep文件保持空目录
                gitkeep = dir_path / '.gitkeep'
                if not gitkeep.exists():
                    gitkeep.touch()
        
        def path_analysis_demo(self):
            """路径分析演示"""
            print("=== pathlib 路径分析演示 ===")
            
            # 创建示例路径
            sample_paths = [
                "D:/自制软件/OAM/LG_output/LG_1.000,0.50pi(1,0)_w_2.540mm_phase.bmp",
                "/home/user/projects/oam/data/measurements.csv",
                "config/default_config.json",
                "..\\backup\\config_backup_20231201.zip"
            ]
            
            for path_str in sample_paths:
                path = Path(path_str)
                
                print(f"\n路径: {path}")
                print(f"  绝对路径: {path.resolve() if path.exists() else '路径不存在'}")
                print(f"  父目录: {path.parent}")
                print(f"  文件名: {path.name}")
                print(f"  文件干名: {path.stem}")
                print(f"  扩展名: {path.suffix}")
                print(f"  所有扩展名: {path.suffixes}")
                print(f"  是否绝对路径: {path.is_absolute()}")
                print(f"  是否存在: {path.exists()}")
                
                if path.exists():
                    print(f"  是否为文件: {path.is_file()}")
                    print(f"  是否为目录: {path.is_dir()}")
                    
                    if path.is_file():
                        stat = path.stat()
                        print(f"  文件大小: {stat.st_size} 字节")
                        print(f"  修改时间: {time.ctime(stat.st_mtime)}")
        
        def advanced_path_operations(self):
            """高级路径操作"""
            print("\n=== 高级路径操作演示 ===")
            
            # 路径拼接的多种方式
            base_path = Path("LG_output")
            
            # 方式1: / 操作符（推荐）
            file_path1 = base_path / "phase_maps" / "LG_mode_1_0.bmp"
            
            # 方式2: joinpath方法
            file_path2 = base_path.joinpath("holograms", "hologram_1_0.bmp")
            
            # 方式3: 动态拼接
            mode_params = {"l": 1, "p": 0, "waist_mm": 2.54}
            filename = f"LG_l{mode_params['l']}_p{mode_params['p']}_w{mode_params['waist_mm']:.1f}mm.bmp"
            file_path3 = base_path / "dynamic" / filename
            
            print(f"拼接路径1: {file_path1}")
            print(f"拼接路径2: {file_path2}")
            print(f"动态路径3: {file_path3}")
            
            # 路径变换
            original_path = Path("LG_output/phase_maps/LG_mode_1_0.bmp")
            
            # 改变扩展名
            png_path = original_path.with_suffix('.png')
            tiff_path = original_path.with_suffix('.tiff')
            
            # 改变文件名
            hologram_path = original_path.with_name('hologram_1_0.bmp')
            
            # 改变目录
            new_dir_path = original_path.with_name(original_path.name).parent.parent / "holograms" / original_path.name
            
            print(f"\n原始路径: {original_path}")
            print(f"PNG格式: {png_path}")
            print(f"TIFF格式: {tiff_path}")
            print(f"全息图路径: {hologram_path}")
            print(f"新目录路径: {new_dir_path}")
            
            # 相对路径计算
            current_file = Path("config/user_settings.json")
            target_file = Path("LG_output/results/latest.bmp")
            
            try:
                relative_path = target_file.relative_to(current_file.parent)
                print(f"\n从 {current_file.parent} 到 {target_file} 的相对路径: {relative_path}")
            except ValueError as e:
                print(f"无法计算相对路径: {e}")
        
        def file_search_and_filtering(self):
            """文件搜索与过滤"""
            print("\n=== 文件搜索与过滤演示 ===")
            
            # 创建一些示例文件用于演示
            test_dir = self.project_root / "test_search"
            test_dir.mkdir(exist_ok=True)
            
            # 创建测试文件
            test_files = [
                "LG_1.000,0.50pi(1,0)_w_2.540mm_phase.bmp",
                "LG_1.000,0.50pi(1,0)_w_2.540mm_hologram.bmp",
                "LG_0.707,0.25pi(-1,0)_w_3.200mm_phase.png",
                "config_backup_20231201.json",
                "measurement_data_20231201.csv",
                "log_20231201.txt",
                "readme.md"
            ]
            
            for filename in test_files:
                (test_dir / filename).touch()
            
            # 1. 基础文件搜索
            print("1. 所有文件:")
            for file_path in test_dir.iterdir():
                if file_path.is_file():
                    print(f"  {file_path.name}")
            
            # 2. 使用glob模式搜索
            print("\n2. BMP文件 (*.bmp):")
            for bmp_file in test_dir.glob("*.bmp"):
                print(f"  {bmp_file.name}")
            
            print("\n3. 相位图文件 (*phase*):")
            for phase_file in test_dir.glob("*phase*"):
                print(f"  {phase_file.name}")
            
            # 3. 递归搜索
            print("\n4. 递归搜索所有图像文件:")
            image_extensions = ['*.bmp', '*.png', '*.jpg', '*.tiff']
            for pattern in image_extensions:
                for img_file in test_dir.rglob(pattern):
                    print(f"  {img_file.relative_to(test_dir)}")
            
            # 4. 高级过滤
            def filter_lg_files(directory: Path) -> Iterator[Path]:
                """过滤LG模式文件"""
                for file_path in directory.iterdir():
                    if (file_path.is_file() and 
                        file_path.name.startswith('LG_') and 
                        file_path.suffix.lower() in ['.bmp', '.png']):
                        yield file_path
            
            print("\n5. LG模式文件:")
            for lg_file in filter_lg_files(test_dir):
                print(f"  {lg_file.name}")
            
            # 5. 按日期过滤
            def filter_by_date(directory: Path, days_old: int = 7) -> Iterator[Path]:
                """过滤指定天数内的文件"""
                cutoff_time = time.time() - (days_old * 24 * 3600)
                
                for file_path in directory.iterdir():
                    if file_path.is_file() and file_path.stat().st_mtime > cutoff_time:
                        yield file_path
            
            print(f"\n6. 最近7天的文件:")
            for recent_file in filter_by_date(test_dir):
                print(f"  {recent_file.name}")
            
            # 清理测试文件
            shutil.rmtree(test_dir)
        
        def file_operations_with_safety(self):
            """安全的文件操作"""
            print("\n=== 安全文件操作演示 ===")
            
            # 创建测试环境
            test_dir = self.project_root / "test_operations"
            test_dir.mkdir(exist_ok=True)
            
            # 创建源文件
            source_file = test_dir / "source.txt"
            source_file.write_text("这是测试内容\n第二行内容", encoding='utf-8')
            
            # 1. 安全复制文件
            def safe_copy_file(source: Path, destination: Path, 
                             backup_existing: bool = True) -> bool:
                """安全复制文件"""
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 如果目标文件存在且需要备份
                    if destination.exists() and backup_existing:
                        backup_path = destination.with_suffix(
                            destination.suffix + f'.backup_{int(time.time())}'
                        )
                        shutil.copy2(destination, backup_path)
                        print(f"已备份现有文件到: {backup_path}")
                    
                    shutil.copy2(source, destination)
                    print(f"成功复制: {source} -> {destination}")
                    return True
                    
                except Exception as e:
                    print(f"复制失败: {e}")
                    return False
            
            # 测试安全复制
            dest_file = test_dir / "backup" / "destination.txt"
            safe_copy_file(source_file, dest_file)
            
            # 2. 安全移动文件
            def safe_move_file(source: Path, destination: Path) -> bool:
                """安全移动文件"""
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 检查目标是否已存在
                    if destination.exists():
                        counter = 1
                        stem = destination.stem
                        suffix = destination.suffix
                        
                        while destination.exists():
                            new_name = f"{stem}_{counter}{suffix}"
                            destination = destination.with_name(new_name)
                            counter += 1
                        
                        print(f"目标文件已存在，重命名为: {destination.name}")
                    
                    shutil.move(str(source), str(destination))
                    print(f"成功移动: {source} -> {destination}")
                    return True
                    
                except Exception as e:
                    print(f"移动失败: {e}")
                    return False
            
            # 3. 批量重命名
            def batch_rename_files(directory: Path, pattern: str, 
                                 replacement: str, dry_run: bool = True) -> List[tuple]:
                """批量重命名文件"""
                rename_operations = []
                
                for file_path in directory.glob(pattern):
                    if file_path.is_file():
                        new_name = file_path.name.replace(pattern.replace('*', ''), replacement)
                        new_path = file_path.with_name(new_name)
                        
                        rename_operations.append((file_path, new_path))
                        
                        if not dry_run:
                            try:
                                file_path.rename(new_path)
                                print(f"重命名: {file_path.name} -> {new_path.name}")
                            except Exception as e:
                                print(f"重命名失败 {file_path.name}: {e}")
                        else:
                            print(f"[试运行] 将重命名: {file_path.name} -> {new_name}")
                
                return rename_operations
            
            # 4. 磁盘空间检查
            def check_disk_space(path: Path, required_space_mb: float = 100) -> bool:
                """检查磁盘空间"""
                try:
                    stat = shutil.disk_usage(path)
                    free_space_mb = stat.free / (1024 * 1024)
                    
                    print(f"磁盘空间信息 ({path}):")
                    print(f"  总空间: {stat.total / (1024**3):.1f} GB")
                    print(f"  已用空间: {stat.used / (1024**3):.1f} GB")
                    print(f"  可用空间: {free_space_mb:.1f} MB")
                    
                    if free_space_mb < required_space_mb:
                        print(f"警告: 可用空间不足! 需要 {required_space_mb} MB")
                        return False
                    
                    return True
                    
                except Exception as e:
                    print(f"检查磁盘空间失败: {e}")
                    return False
            
            # 测试磁盘空间检查
            check_disk_space(test_dir)
            
            # 清理
            shutil.rmtree(test_dir)
        
        def temporary_file_management(self):
            """临时文件管理"""
            print("\n=== 临时文件管理演示 ===")
            
            # 1. 使用tempfile模块
            import tempfile
            
            # 临时目录
            with tempfile.TemporaryDirectory(prefix='oam_temp_') as temp_dir:
                temp_path = Path(temp_dir)
                print(f"临时目录: {temp_path}")
                
                # 在临时目录中创建文件
                temp_file = temp_path / "calculation_result.txt"
                temp_file.write_text("临时计算结果", encoding='utf-8')
                
                print(f"临时文件: {temp_file}")
                print(f"临时文件存在: {temp_file.exists()}")
                
                # 临时目录会在with块结束时自动删除
            
            print(f"退出with块后，临时目录已删除: {not temp_path.exists()}")
            
            # 2. 自定义临时文件管理
            class TempFileManager:
                """临时文件管理器"""
                
                def __init__(self, base_dir: Path = None):
                    self.base_dir = base_dir or Path.cwd() / "temp"
                    self.temp_files = set()
                    self.base_dir.mkdir(exist_ok=True)
                
                def create_temp_file(self, prefix: str = "temp_", 
                                   suffix: str = ".tmp") -> Path:
                    """创建临时文件"""
                    timestamp = int(time.time() * 1000000)  # 微秒时间戳
                    filename = f"{prefix}{timestamp}{suffix}"
                    temp_file = self.base_dir / filename
                    
                    temp_file.touch()
                    self.temp_files.add(temp_file)
                    
                    return temp_file
                
                def cleanup(self):
                    """清理所有临时文件"""
                    for temp_file in self.temp_files.copy():
                        try:
                            if temp_file.exists():
                                temp_file.unlink()
                                print(f"已删除临时文件: {temp_file.name}")
                            self.temp_files.remove(temp_file)
                        except Exception as e:
                            print(f"删除临时文件失败 {temp_file}: {e}")
                
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.cleanup()
            
            # 使用自定义临时文件管理器
            with TempFileManager() as temp_mgr:
                # 创建一些临时文件
                temp1 = temp_mgr.create_temp_file("phase_", ".bmp")
                temp2 = temp_mgr.create_temp_file("hologram_", ".bmp")
                
                print(f"创建临时文件: {temp1.name}, {temp2.name}")
                
                # 临时文件会在退出时自动清理
        
        def generate_safe_filename(self, coeffs: List[complex], l_list: List[int], 
                                 p_list: List[int], waist: float, 
                                 prefix: str = "LG_", timestamp: bool = False) -> str:
            """生成安全的文件名"""
            # 构建模式部分
            mode_parts = []
            for c, p, l in zip(coeffs, p_list, l_list):
                amp = abs(c)
                phase_over_pi = np.angle(c) / np.pi if hasattr(c, '__complex__') else 0
                mode_parts.append(f"{amp:.3f},{phase_over_pi:.2f}pi({p},{l})")
            
            # 构建光腰部分
            waist_mm = waist * 1000
            waist_part = f"w_{waist_mm:.3f}mm"
            
            # 基础文件名
            base_name = f"{prefix}{'+'.join(mode_parts)}_{waist_part}"
            
            # 添加时间戳
            if timestamp:
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name += f"_{ts}"
            
            # 清理文件名中的非法字符
            import re
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
            safe_name = re.sub(r'\s+', '_', safe_name)
            
            # 限制长度
            if len(safe_name) > 200:
                safe_name = safe_name[:200]
            
            return safe_name
        
        def run_demonstrations(self):
            """运行所有演示"""
            self.path_analysis_demo()
            self.advanced_path_operations()
            self.file_search_and_filtering()
            self.file_operations_with_safety()
            self.temporary_file_management()
            
            # 测试文件名生成
            print("\n=== 安全文件名生成演示 ===")
            import numpy as np
            
            test_coeffs = [1+0j, 1+0j]
            test_l = [1, -1]
            test_p = [0, 0]
            test_waist = 0.00254
            
            safe_filename = self.generate_safe_filename(
                test_coeffs, test_l, test_p, test_waist, timestamp=True
            )
            print(f"生成的安全文件名: {safe_filename}")
    
    return ProjectFileManager

def demonstrate_file_io_best_practices():
    """文件IO最佳实践演示"""
    
    class FileIOBestPractices:
        """文件IO最佳实践示例"""
        
        @staticmethod
        def safe_file_read(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
            """安全读取文件"""
            try:
                if not file_path.exists():
                    print(f"文件不存在: {file_path}")
                    return None
                
                if not file_path.is_file():
                    print(f"路径不是文件: {file_path}")
                    return None
                
                # 检查文件大小
                file_size = file_path.stat().st_size
                if file_size > 100 * 1024 * 1024:  # 100MB
                    print(f"警告: 文件过大 ({file_size / 1024 / 1024:.1f} MB)")
                    response = input("是否继续读取? (y/n): ")
                    if response.lower() != 'y':
                        return None
                
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                print(f"成功读取文件: {file_path} ({len(content)} 字符)")
                return content
                
            except UnicodeDecodeError as e:
                print(f"编码错误: {e}")
                # 尝试其他编码
                for alt_encoding in ['gbk', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=alt_encoding) as f:
                            content = f.read()
                        print(f"使用 {alt_encoding} 编码成功读取")
                        return content
                    except:
                        continue
                return None
                
            except Exception as e:
                print(f"读取文件失败: {e}")
                return None
        
        @staticmethod
        def safe_file_write(file_path: Path, content: str, 
                          encoding: str = 'utf-8', backup: bool = True) -> bool:
            """安全写入文件"""
            try:
                # 确保目录存在
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 备份现有文件
                if file_path.exists() and backup:
                    backup_path = file_path.with_suffix(
                        file_path.suffix + f'.backup_{int(time.time())}'
                    )
                    shutil.copy2(file_path, backup_path)
                    print(f"已备份到: {backup_path}")
                
                # 原子写入（先写临时文件，再重命名）
                temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
                
                with open(temp_path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                # 原子替换
                temp_path.replace(file_path)
                print(f"成功写入文件: {file_path}")
                return True
                
            except Exception as e:
                print(f"写入文件失败: {e}")
                # 清理临时文件
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
                return False
        
        @staticmethod
        def chunked_file_read(file_path: Path, chunk_size: int = 8192) -> Iterator[str]:
            """分块读取大文件"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                print(f"分块读取失败: {e}")
        
        @staticmethod
        def binary_file_operations():
            """二进制文件操作示例"""
            import struct
            
            # 创建测试目录
            test_dir = Path("test_binary")
            test_dir.mkdir(exist_ok=True)
            
            # 写入二进制数据
            binary_file = test_dir / "data.bin"
            
            # 示例：保存图像尺寸和像素数据
            width, height = 1920, 1152
            pixel_data = bytes(range(256)) * ((width * height) // 256 + 1)
            pixel_data = pixel_data[:width * height]
            
            with open(binary_file, 'wb') as f:
                # 写入头部信息
                f.write(struct.pack('<II', width, height))  # 小端序，两个unsigned int
                # 写入像素数据
                f.write(pixel_data)
            
            print(f"写入二进制文件: {binary_file}")
            print(f"文件大小: {binary_file.stat().st_size} 字节")
            
            # 读取二进制数据
            with open(binary_file, 'rb') as f:
                # 读取头部
                header = f.read(8)  # 两个4字节整数
                w, h = struct.unpack('<II', header)
                
                # 读取像素数据
                pixels = f.read(w * h)
                
                print(f"读取图像尺寸: {w} x {h}")
                print(f"像素数据长度: {len(pixels)} 字节")
            
            # 清理
            shutil.rmtree(test_dir)
        
        @classmethod
        def run_demonstrations(cls):
            """运行所有演示"""
            # 创建测试文件
            test_dir = Path("test_io")
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / "test.txt"
            
            # 测试安全写入
            content = "这是测试内容\n包含中文字符\n第三行"
            cls.safe_file_write(test_file, content)
            
            # 测试安全读取
            read_content = cls.safe_file_read(test_file)
            print(f"读取内容预览: {read_content[:50]}...")
            
            # 测试分块读取
            print("\n分块读取演示:")
            for i, chunk in enumerate(cls.chunked_file_read(test_file)):
                print(f"块 {i+1}: {len(chunk)} 字符")
                if i >= 2:  # 只显示前3块
                    break
            
            # 测试二进制操作
            print("\n二进制文件操作:")
            cls.binary_file_operations()
            
            # 清理
            shutil.rmtree(test_dir)
    
    return FileIOBestPractices

# 使用示例
def demo_pathlib_and_file_io():
    """演示pathlib和文件IO"""
    print("=== pathlib 和文件IO演示 ===")
    
    # 文件管理器演示
    file_manager_class = comprehensive_pathlib_usage()
    file_manager = file_manager_class()
    file_manager.run_demonstrations()
    
    print("\n" + "="*50)
    
    # 文件IO最佳实践演示
    io_practices_class = demonstrate_file_io_best_practices()
    io_practices = io_practices_class()
    io_practices.run_demonstrations()
    
    return file_manager, io_practices
```

### 10.2 文件监控与自动化处理

#### 10.2.1 watchdog文件监控实现
```python
import time
import threading
from queue import Queue
from typing import Callable, Dict, Any

def file_monitoring_system():
    """文件监控系统"""
    
    class FileMonitor:
        """文件监控器 - 监控OAM输出文件的变化"""
        
        def __init__(self, watch_directory: Path):
            self.watch_directory = Path(watch_directory)
            self.watch_directory.mkdir(exist_ok=True)
            
            self.is_monitoring = False
            self.event_queue = Queue()
            self.event_handlers = {}
            
            # 文件状态缓存
            self.file_states = {}
            self.last_scan_time = time.time()
        
        def add_event_handler(self, event_type: str, handler: Callable):
            """添加事件处理器"""
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(handler)
        
        def scan_directory(self):
            """扫描目录变化"""
            current_states = {}
            
            try:
                for file_path in self.watch_directory.rglob('*'):
                    if file_path.is_file():
                        stat = file_path.stat()
                        current_states[str(file_path)] = {
                            'size': stat.st_size,
                            'mtime': stat.st_mtime,
                            'exists': True
                        }
                
                # 检测变化
                self._detect_changes(current_states)
                self.file_states = current_states
                
            except Exception as e:
                print(f"扫描目录失败: {e}")
        
        def _detect_changes(self, current_states: Dict[str, Dict]):
            """检测文件变化"""
            # 新文件
            for file_path, state in current_states.items():
                if file_path not in self.file_states:
                    self._trigger_event('file_created', {
                        'path': file_path,
                        'size': state['size']
                    })
            
            # 修改的文件
            for file_path, state in current_states.items():
                if file_path in self.file_states:
                    old_state = self.file_states[file_path]
                    if (state['mtime'] > old_state['mtime'] or 
                        state['size'] != old_state['size']):
                        self._trigger_event('file_modified', {
                            'path': file_path,
                            'old_size': old_state['size'],
                            'new_size': state['size']
                        })
            
            # 删除的文件
            for file_path in self.file_states:
                if file_path not in current_states:
                    self._trigger_event('file_deleted', {
                        'path': file_path
                    })
        
        def _trigger_event(self, event_type: str, data: Dict[str, Any]):
            """触发事件"""
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        handler(data)
                    except Exception as e:
                        print(f"事件处理器错误: {e}")
        
        def start_monitoring(self, interval: float = 1.0):
            """开始监控"""
            if self.is_monitoring:
                return
            
            self.is_monitoring = True
            
            def monitor_loop():
                while self.is_monitoring:
                    self.scan_directory()
                    time.sleep(interval)
            
            self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            self.monitor_thread.start()
            print(f"开始监控目录: {self.watch_directory}")
        
        def stop_monitoring(self):
            """停止监控"""
            self.is_monitoring = False
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=5)
            print("停止文件监控")
    
    class AutoProcessor:
        """自动处理器 - 处理检测到的文件变化"""
        
        def __init__(self, output_dir: Path):
            self.output_dir = Path(output_dir)
            self.processing_queue = Queue()
            self.is_processing = False
        
        def on_file_created(self, event_data: Dict[str, Any]):
            """处理文件创建事件"""
            file_path = Path(event_data['path'])
            print(f"检测到新文件: {file_path.name}")
            
            # 检查是否是图像文件
            if file_path.suffix.lower() in ['.bmp', '.png', '.tiff']:
                self.processing_queue.put(('process_image', file_path))
            elif file_path.suffix.lower() in ['.json']:
                self.processing_queue.put(('process_config', file_path))
        
        def on_file_modified(self, event_data: Dict[str, Any]):
            """处理文件修改事件"""
            file_path = Path(event_data['path'])
            old_size = event_data['old_size']
            new_size = event_data['new_size']
            
            print(f"文件已修改: {file_path.name} ({old_size} -> {new_size} 字节)")
        
        def on_file_deleted(self, event_data: Dict[str, Any]):
            """处理文件删除事件"""
            file_path = Path(event_data['path'])
            print(f"文件已删除: {file_path}")
        
        def process_image_file(self, file_path: Path):
            """处理图像文件"""
            try:
                # 模拟图像处理
                print(f"处理图像文件: {file_path.name}")
                
                # 检查文件完整性
                file_size = file_path.stat().st_size
                if file_size == 0:
                    print(f"警告: 图像文件为空 {file_path.name}")
                    return
                
                # 生成缩略图
                thumbnail_dir = self.output_dir / "thumbnails"
                thumbnail_dir.mkdir(exist_ok=True)
                
                thumbnail_path = thumbnail_dir / f"thumb_{file_path.name}"
                
                # 这里可以调用实际的图像处理代码
                # 例如使用PIL生成缩略图
                print(f"生成缩略图: {thumbnail_path}")
                
                # 更新元数据
                self.update_image_metadata(file_path)
                
            except Exception as e:
                print(f"处理图像文件失败: {e}")
        
        def process_config_file(self, file_path: Path):
            """处理配置文件"""
            try:
                print(f"处理配置文件: {file_path.name}")
                
                # 验证JSON格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 备份配置文件
                backup_dir = self.output_dir / "config_backup"
                backup_dir.mkdir(exist_ok=True)
                
                timestamp = int(time.time())
                backup_path = backup_dir / f"{file_path.stem}_{timestamp}.json"
                shutil.copy2(file_path, backup_path)
                
                print(f"配置文件已备份: {backup_path}")
                
            except json.JSONDecodeError as e:
                print(f"配置文件JSON格式错误: {e}")
            except Exception as e:
                print(f"处理配置文件失败: {e}")
        
        def update_image_metadata(self, image_path: Path):
            """更新图像元数据"""
            metadata_file = self.output_dir / "image_metadata.json"
            
            # 读取现有元数据
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # 添加新图像信息
            stat = image_path.stat()
            metadata[str(image_path)] = {
                'name': image_path.name,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'processed': time.time()
            }
            
            # 保存元数据
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        def start_processing(self):
            """开始处理队列"""
            if self.is_processing:
                return
            
            self.is_processing = True
            
            def process_loop():
                while self.is_processing:
                    try:
                        if not self.processing_queue.empty():
                            task_type, file_path = self.processing_queue.get(timeout=1)
                            
                            if task_type == 'process_image':
                                self.process_image_file(file_path)
                            elif task_type == 'process_config':
                                self.process_config_file(file_path)
                            
                            self.processing_queue.task_done()
                        else:
                            time.sleep(0.1)
                            
                    except Exception as e:
                        print(f"处理队列错误: {e}")
                        time.sleep(1)
            
            self.process_thread = threading.Thread(target=process_loop, daemon=True)
            self.process_thread.start()
            print("自动处理器已启动")
        
        def stop_processing(self):
            """停止处理"""
            self.is_processing = False
            if hasattr(self, 'process_thread'):
                self.process_thread.join(timeout=5)
            print("自动处理器已停止")
    
    return FileMonitor, AutoProcessor

# 使用示例
def demo_file_monitoring():
    """演示文件监控系统"""
    print("=== 文件监控系统演示 ===")
    
    # 创建测试目录
    watch_dir = Path("test_monitoring")
    watch_dir.mkdir(exist_ok=True)
    
    output_dir = Path("processed_output")
    output_dir.mkdir(exist_ok=True)
    
    # 获取类
    FileMonitor, AutoProcessor = file_monitoring_system()
    
    # 创建监控器和处理器
    monitor = FileMonitor(watch_dir)
    processor = AutoProcessor(output_dir)
    
    # 设置事件处理器
    monitor.add_event_handler('file_created', processor.on_file_created)
    monitor.add_event_handler('file_modified', processor.on_file_modified)
    monitor.add_event_handler('file_deleted', processor.on_file_deleted)
    
    # 启动监控和处理
    monitor.start_monitoring(interval=0.5)
    processor.start_processing()
    
    try:
        print("文件监控系统运行中，创建一些测试文件...")
        
        # 创建测试文件
        test_files = [
            "test_image.bmp",
            "config.json",
            "data.txt"
        ]
        
        for filename in test_files:
            test_file = watch_dir / filename
            
            if filename.endswith('.json'):
                test_file.write_text('{"test": "data"}', encoding='utf-8')
            else:
                test_file.write_text("测试内容", encoding='utf-8')
            
            print(f"创建文件: {filename}")
            time.sleep(1)
        
        # 修改文件
        time.sleep(1)
        (watch_dir / "data.txt").write_text("修改后的内容", encoding='utf-8')
        print("修改文件: data.txt")
        
        # 删除文件
        time.sleep(1)
        (watch_dir / "data.txt").unlink()
        print("删除文件: data.txt")
        
        # 等待处理完成
        time.sleep(2)
        
    finally:
        # 停止监控
        monitor.stop_monitoring()
        processor.stop_processing()
        
        # 清理测试目录
        shutil.rmtree(watch_dir)
        shutil.rmtree(output_dir)
        
        print("文件监控演示完成")

---

## 第11章：Excel 处理与批量数据管道

### 11.1 openpyxl 深度应用

#### 11.1.1 Excel文件的读取、写入与批量处理
```python
import openpyxl
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.chart import ScatterChart, Reference, Series
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import re
from dataclasses import dataclass

def comprehensive_excel_processing():
    """Excel处理完整教学"""
    
    @dataclass
    class LGModeParameters:
        """LG模式参数数据类"""
        l: int
        p: int
        amplitude: float
        phase: float
        waist: float
        enabled: bool = True
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'l': self.l,
                'p': self.p,
                'amplitude': self.amplitude,
                'phase': self.phase,
                'waist': self.waist,
                'enabled': self.enabled
            }
    
    class ExcelProcessor:
        """Excel处理器 - 处理OAM项目的批量参数"""
        
        def __init__(self):
            self.supported_columns = {
                'l': ['l', 'l_value', '角量子数', 'azimuthal'],
                'p': ['p', 'p_value', '径向量子数', 'radial'],
                'amplitude': ['amplitude', 'amp', '幅度', 'magnitude'],
                'phase': ['phase', '相位', 'phase_pi', '相位/π'],
                'waist': ['waist', 'w', '光腰', 'beam_waist', '光腰半径'],
                'enabled': ['enabled', 'enable', '启用', 'active']
            }
        
        def detect_column_mapping(self, worksheet) -> Dict[str, str]:
            """智能检测列映射"""
            header_row = 1
            column_mapping = {}
            
            # 读取表头
            headers = []
            for col in range(1, worksheet.max_column + 1):
                cell_value = worksheet.cell(row=header_row, column=col).value
                if cell_value:
                    headers.append((col, str(cell_value).strip().lower()))
            
            # 匹配列名
            for param, possible_names in self.supported_columns.items():
                for col_idx, header in headers:
                    if any(name.lower() in header for name in possible_names):
                        column_mapping[param] = get_column_letter(col_idx)
                        break
            
            return column_mapping
        
        def read_excel_parameters(self, file_path: Path) -> List[LGModeParameters]:
            """从Excel读取参数"""
            try:
                workbook = load_workbook(file_path, data_only=True)
                worksheet = workbook.active
                
                # 检测列映射
                column_mapping = self.detect_column_mapping(worksheet)
                print(f"检测到的列映射: {column_mapping}")
                
                if not column_mapping:
                    raise ValueError("未能识别任何有效的列")
                
                parameters = []
                
                # 从第2行开始读取数据（第1行是表头）
                for row in range(2, worksheet.max_row + 1):
                    try:
                        # 读取各参数
                        l_val = self._get_cell_value(worksheet, row, column_mapping.get('l'), int, 0)
                        p_val = self._get_cell_value(worksheet, row, column_mapping.get('p'), int, 0)
                        amp_val = self._get_cell_value(worksheet, row, column_mapping.get('amplitude'), float, 1.0)
                        phase_val = self._get_cell_value(worksheet, row, column_mapping.get('phase'), float, 0.0)
                        waist_val = self._get_cell_value(worksheet, row, column_mapping.get('waist'), float, 0.00254)
                        enabled_val = self._get_cell_value(worksheet, row, column_mapping.get('enabled'), bool, True)
                        
                        # 跳过空行
                        if l_val is None and p_val is None:
                            continue
                        
                        # 创建参数对象
                        param = LGModeParameters(
                            l=l_val or 0,
                            p=p_val or 0,
                            amplitude=amp_val or 1.0,
                            phase=phase_val or 0.0,
                            waist=waist_val or 0.00254,
                            enabled=enabled_val
                        )
                        
                        parameters.append(param)
                        
                    except Exception as e:
                        print(f"读取第{row}行数据时出错: {e}")
                        continue
                
                workbook.close()
                print(f"成功读取 {len(parameters)} 组参数")
                return parameters
                
            except FileNotFoundError:
                raise FileNotFoundError(f"Excel文件不存在: {file_path}")
            except Exception as e:
                raise Exception(f"读取Excel文件失败: {e}")
        
        def _get_cell_value(self, worksheet, row: int, column: str, 
                          target_type: type, default_value: Any) -> Any:
            """安全获取单元格值"""
            if not column:
                return default_value
            
            try:
                cell_value = worksheet[f"{column}{row}"].value
                
                if cell_value is None:
                    return default_value
                
                if target_type == bool:
                    if isinstance(cell_value, bool):
                        return cell_value
                    elif isinstance(cell_value, str):
                        return cell_value.lower() in ['true', '1', 'yes', 'on', '是', '启用']
                    else:
                        return bool(cell_value)
                
                elif target_type == int:
                    return int(float(cell_value))  # 处理Excel中的数字格式
                
                elif target_type == float:
                    return float(cell_value)
                
                else:
                    return target_type(cell_value)
                    
            except (ValueError, TypeError) as e:
                print(f"转换单元格值失败 ({column}{row}): {e}")
                return default_value
        
        def write_excel_parameters(self, file_path: Path, 
                                 parameters: List[LGModeParameters],
                                 include_results: bool = False) -> None:
            """写入参数到Excel"""
            try:
                workbook = Workbook()
                worksheet = workbook.active
                worksheet.title = "LG模式参数"
                
                # 设置表头
                headers = ['序号', 'l值', 'p值', '幅度', '相位/π', '光腰(mm)', '启用']
                if include_results:
                    headers.extend(['文件名', '生成状态', '文件大小(KB)', '生成时间'])
                
                for col, header in enumerate(headers, 1):
                    cell = worksheet.cell(row=1, column=col)
                    cell.value = header
                    cell.font = Font(bold=True)
                    cell.alignment = Alignment(horizontal='center')
                    cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
                
                # 写入数据
                for row, param in enumerate(parameters, 2):
                    worksheet.cell(row=row, column=1, value=row-1)  # 序号
                    worksheet.cell(row=row, column=2, value=param.l)
                    worksheet.cell(row=row, column=3, value=param.p)
                    worksheet.cell(row=row, column=4, value=param.amplitude)
                    worksheet.cell(row=row, column=5, value=param.phase)
                    worksheet.cell(row=row, column=6, value=param.waist * 1000)  # 转换为mm
                    worksheet.cell(row=row, column=7, value="是" if param.enabled else "否")
                    
                    if include_results:
                        # 模拟结果数据
                        worksheet.cell(row=row, column=8, value=f"LG_{param.l}_{param.p}.bmp")
                        worksheet.cell(row=row, column=9, value="已生成" if param.enabled else "跳过")
                        worksheet.cell(row=row, column=10, value=2048 if param.enabled else 0)
                        worksheet.cell(row=row, column=11, value="2023-12-01 12:00:00")
                
                # 自动调整列宽
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 30)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # 添加边框
                thin_border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
                
                for row in worksheet.iter_rows(min_row=1, max_row=len(parameters)+1,
                                             min_col=1, max_col=len(headers)):
                    for cell in row:
                        cell.border = thin_border
                
                # 保存文件
                file_path.parent.mkdir(parents=True, exist_ok=True)
                workbook.save(file_path)
                print(f"Excel文件已保存: {file_path}")
                
            except Exception as e:
                raise Exception(f"写入Excel文件失败: {e}")
        
        def create_batch_template(self, file_path: Path, num_rows: int = 10) -> None:
            """创建批量处理模板"""
            try:
                workbook = Workbook()
                worksheet = workbook.active
                worksheet.title = "批量参数模板"
                
                # 设置表头和说明
                headers_info = [
                    ('A1', 'l值', '角量子数，整数，可为负'),
                    ('B1', 'p值', '径向量子数，非负整数'),
                    ('C1', '幅度', '模式幅度，浮点数，通常0-1'),
                    ('D1', '相位/π', '相位除以π，浮点数'),
                    ('E1', '光腰(mm)', '光腰半径，毫米，浮点数'),
                    ('F1', '启用', '是否启用此模式，是/否')
                ]
                
                # 写入表头
                for col_addr, header, description in headers_info:
                    cell = worksheet[col_addr]
                    cell.value = header
                    cell.font = Font(bold=True)
                    cell.alignment = Alignment(horizontal='center')
                    cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
                    
                    # 添加批注
                    from openpyxl.comments import Comment
                    cell.comment = Comment(description, "系统")
                
                # 添加示例数据
                example_data = [
                    (1, 0, 1.0, 0.0, 2.54, '是'),
                    (-1, 0, 1.0, 0.5, 2.54, '是'),
                    (2, 1, 0.707, 0.25, 3.2, '是'),
                    (-2, 1, 0.707, 0.75, 3.2, '否'),
                ]
                
                for row, data in enumerate(example_data, 2):
                    for col, value in enumerate(data, 1):
                        worksheet.cell(row=row, column=col, value=value)
                
                # 添加空行供用户填写
                for row in range(len(example_data) + 2, num_rows + 2):
                    worksheet.cell(row=row, column=1, value=0)  # 默认l值
                    worksheet.cell(row=row, column=2, value=0)  # 默认p值
                    worksheet.cell(row=row, column=3, value=1.0)  # 默认幅度
                    worksheet.cell(row=row, column=4, value=0.0)  # 默认相位
                    worksheet.cell(row=row, column=5, value=2.54)  # 默认光腰
                    worksheet.cell(row=row, column=6, value='是')  # 默认启用
                
                # 设置数据验证
                from openpyxl.worksheet.datavalidation import DataValidation
                
                # 启用列的数据验证
                dv = DataValidation(type="list", formula1='"是,否"', allow_blank=False)
                dv.error = '请选择"是"或"否"'
                dv.errorTitle = '输入错误'
                worksheet.add_data_validation(dv)
                dv.add(f'F2:F{num_rows + 1}')
                
                # 自动调整列宽
                column_widths = [8, 8, 12, 12, 15, 10]
                for i, width in enumerate(column_widths, 1):
                    worksheet.column_dimensions[get_column_letter(i)].width = width
                
                # 添加说明工作表
                info_sheet = workbook.create_sheet("使用说明")
                instructions = [
                    "OAM全息图批量生成参数模板使用说明",
                    "",
                    "1. 参数说明：",
                    "   - l值：角量子数，决定螺旋相位的旋向和圈数",
                    "   - p值：径向量子数，决定径向节点数",
                    "   - 幅度：该模式的相对强度",
                    "   - 相位/π：相位偏移除以π的值",
                    "   - 光腰(mm)：高斯光束的腰半径",
                    "   - 启用：选择是否生成此模式的全息图",
                    "",
                    "2. 填写注意事项：",
                    "   - l值可以为正负整数",
                    "   - p值必须为非负整数",
                    "   - 幅度建议在0-1之间",
                    "   - 相位/π的范围通常为0-2",
                    "   - 光腰单位为毫米",
                    "",
                    "3. 批量生成：",
                    "   - 保存此文件后，在软件中选择此文件",
                    "   - 软件将按照启用的行生成对应的全息图",
                    "   - 生成结果将保存到指定目录"
                ]
                
                for row, instruction in enumerate(instructions, 1):
                    info_sheet.cell(row=row, column=1, value=instruction)
                    if row == 1:  # 标题
                        info_sheet.cell(row=row, column=1).font = Font(bold=True, size=14)
                
                info_sheet.column_dimensions['A'].width = 50
                
                # 保存文件
                file_path.parent.mkdir(parents=True, exist_ok=True)
                workbook.save(file_path)
                print(f"批量处理模板已创建: {file_path}")
                
            except Exception as e:
                raise Exception(f"创建模板失败: {e}")
        
        def validate_parameters(self, parameters: List[LGModeParameters]) -> Tuple[List[LGModeParameters], List[str]]:
            """验证参数有效性"""
            valid_params = []
            errors = []
            
            for i, param in enumerate(parameters):
                param_errors = []
                
                # 验证l值（可以为任意整数）
                if not isinstance(param.l, int):
                    param_errors.append("l值必须为整数")
                
                # 验证p值（必须为非负整数）
                if not isinstance(param.p, int) or param.p < 0:
                    param_errors.append("p值必须为非负整数")
                
                # 验证幅度
                if not isinstance(param.amplitude, (int, float)) or param.amplitude < 0:
                    param_errors.append("幅度必须为非负数")
                
                # 验证相位
                if not isinstance(param.phase, (int, float)):
                    param_errors.append("相位必须为数值")
                
                # 验证光腰
                if not isinstance(param.waist, (int, float)) or param.waist <= 0:
                    param_errors.append("光腰必须为正数")
                
                if param_errors:
                    errors.append(f"第{i+1}行参数错误: {'; '.join(param_errors)}")
                else:
                    valid_params.append(param)
            
            return valid_params, errors
        
        def generate_statistics_report(self, parameters: List[LGModeParameters], 
                                     file_path: Path) -> None:
            """生成统计报告"""
            try:
                workbook = Workbook()
                
                # 统计工作表
                stats_sheet = workbook.active
                stats_sheet.title = "参数统计"
                
                # 基本统计
                total_params = len(parameters)
                enabled_params = sum(1 for p in parameters if p.enabled)
                unique_l_values = len(set(p.l for p in parameters))
                unique_p_values = len(set(p.p for p in parameters))
                
                avg_amplitude = sum(p.amplitude for p in parameters) / total_params if total_params > 0 else 0
                avg_waist = sum(p.waist for p in parameters) / total_params if total_params > 0 else 0
                
                # 写入统计信息
                stats_data = [
                    ("基本统计", ""),
                    ("总参数数量", total_params),
                    ("启用参数数量", enabled_params),
                    ("禁用参数数量", total_params - enabled_params),
                    ("唯一l值数量", unique_l_values),
                    ("唯一p值数量", unique_p_values),
                    ("平均幅度", f"{avg_amplitude:.3f}"),
                    ("平均光腰(mm)", f"{avg_waist * 1000:.3f}"),
                    ("", ""),
                    ("l值分布", ""),
                ]
                
                # l值分布统计
                l_distribution = {}
                for param in parameters:
                    l_distribution[param.l] = l_distribution.get(param.l, 0) + 1
                
                for l_val, count in sorted(l_distribution.items()):
                    stats_data.append((f"l={l_val}", count))
                
                stats_data.extend([("", ""), ("p值分布", "")])
                
                # p值分布统计
                p_distribution = {}
                for param in parameters:
                    p_distribution[param.p] = p_distribution.get(param.p, 0) + 1
                
                for p_val, count in sorted(p_distribution.items()):
                    stats_data.append((f"p={p_val}", count))
                
                # 写入数据
                for row, (label, value) in enumerate(stats_data, 1):
                    stats_sheet.cell(row=row, column=1, value=label)
                    stats_sheet.cell(row=row, column=2, value=value)
                    
                    # 设置标题样式
                    if label in ["基本统计", "l值分布", "p值分布"]:
                        stats_sheet.cell(row=row, column=1).font = Font(bold=True)
                
                # 调整列宽
                stats_sheet.column_dimensions['A'].width = 20
                stats_sheet.column_dimensions['B'].width = 15
                
                # 详细参数表
                detail_sheet = workbook.create_sheet("详细参数")
                
                headers = ['序号', 'l值', 'p值', '幅度', '相位/π', '光腰(mm)', '启用状态']
                for col, header in enumerate(headers, 1):
                    cell = detail_sheet.cell(row=1, column=col)
                    cell.value = header
                    cell.font = Font(bold=True)
                
                for row, param in enumerate(parameters, 2):
                    detail_sheet.cell(row=row, column=1, value=row-1)
                    detail_sheet.cell(row=row, column=2, value=param.l)
                    detail_sheet.cell(row=row, column=3, value=param.p)
                    detail_sheet.cell(row=row, column=4, value=f"{param.amplitude:.3f}")
                    detail_sheet.cell(row=row, column=5, value=f"{param.phase:.3f}")
                    detail_sheet.cell(row=row, column=6, value=f"{param.waist * 1000:.3f}")
                    detail_sheet.cell(row=row, column=7, value="启用" if param.enabled else "禁用")
                
                # 自动调整列宽
                for column in detail_sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 20)
                    detail_sheet.column_dimensions[column_letter].width = adjusted_width
                
                # 保存文件
                file_path.parent.mkdir(parents=True, exist_ok=True)
                workbook.save(file_path)
                print(f"统计报告已生成: {file_path}")
                
            except Exception as e:
                raise Exception(f"生成统计报告失败: {e}")
    
    return ExcelProcessor, LGModeParameters
```

### 11.2 数据管道设计

#### 11.2.1 批量处理流水线
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from queue import Queue, Empty
import logging
from enum import Enum

def advanced_batch_processing():
    """高级批量处理系统"""
    
    class ProcessingStatus(Enum):
        """处理状态枚举"""
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
        SKIPPED = "skipped"
    
    @dataclass
    class ProcessingTask:
        """处理任务"""
        id: str
        parameters: 'LGModeParameters'
        output_path: Path
        status: ProcessingStatus = ProcessingStatus.PENDING
        error_message: str = ""
        start_time: Optional[float] = None
        end_time: Optional[float] = None
        result_files: List[Path] = None
        
        def __post_init__(self):
            if self.result_files is None:
                self.result_files = []
        
        @property
        def duration(self) -> Optional[float]:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    class BatchProcessor:
        """批量处理器"""
        
        def __init__(self, max_workers: int = None, use_multiprocessing: bool = False):
            self.max_workers = max_workers or multiprocessing.cpu_count()
            self.use_multiprocessing = use_multiprocessing
            self.tasks = []
            self.completed_tasks = []
            self.failed_tasks = []
            
            # 设置日志
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        
        def add_task(self, task: ProcessingTask):
            """添加处理任务"""
            self.tasks.append(task)
            self.logger.info(f"添加任务: {task.id}")
        
        def add_tasks_from_excel(self, excel_file: Path, output_dir: Path):
            """从Excel文件添加任务"""
            try:
                processor = ExcelProcessor()
                parameters = processor.read_excel_parameters(excel_file)
                valid_params, errors = processor.validate_parameters(parameters)
                
                if errors:
                    self.logger.warning(f"参数验证发现 {len(errors)} 个错误")
                    for error in errors:
                        self.logger.warning(error)
                
                for i, param in enumerate(valid_params):
                    if param.enabled:
                        task_id = f"task_{i+1:04d}"
                        task = ProcessingTask(
                            id=task_id,
                            parameters=param,
                            output_path=output_dir / f"{task_id}"
                        )
                        self.add_task(task)
                
                self.logger.info(f"从Excel加载了 {len(valid_params)} 个有效任务")
                
            except Exception as e:
                self.logger.error(f"从Excel加载任务失败: {e}")
                raise
        
        def process_single_task(self, task: ProcessingTask) -> ProcessingTask:
            """处理单个任务"""
            task.start_time = time.time()
            task.status = ProcessingStatus.PROCESSING
            
            try:
                self.logger.info(f"开始处理任务: {task.id}")
                
                # 确保输出目录存在
                task.output_path.mkdir(parents=True, exist_ok=True)
                
                # 模拟全息图生成过程
                self._simulate_hologram_generation(task)
                
                task.status = ProcessingStatus.COMPLETED
                task.end_time = time.time()
                
                self.logger.info(f"任务完成: {task.id} (耗时: {task.duration:.2f}s)")
                
            except Exception as e:
                task.status = ProcessingStatus.FAILED
                task.error_message = str(e)
                task.end_time = time.time()
                
                self.logger.error(f"任务失败: {task.id} - {e}")
            
            return task
        
        def _simulate_hologram_generation(self, task: ProcessingTask):
            """模拟全息图生成（实际项目中会调用真实的生成函数）"""
            import numpy as np
            
            param = task.parameters
            
            # 模拟参数验证
            if abs(param.l) > 20:
                raise ValueError(f"l值过大: {param.l}")
            
            if param.p > 10:
                raise ValueError(f"p值过大: {param.p}")
            
            # 模拟图像生成
            time.sleep(0.1)  # 模拟计算时间
            
            # 生成文件名
            safe_filename = self._generate_filename(param)
            
            # 模拟保存相位图
            phase_file = task.output_path / f"{safe_filename}_phase.bmp"
            phase_file.touch()  # 实际项目中会保存真实图像
            task.result_files.append(phase_file)
            
            # 模拟保存全息图
            hologram_file = task.output_path / f"{safe_filename}_hologram.bmp"
            hologram_file.touch()  # 实际项目中会保存真实图像
            task.result_files.append(hologram_file)
            
            self.logger.debug(f"生成文件: {phase_file.name}, {hologram_file.name}")
        
        def _generate_filename(self, param: 'LGModeParameters') -> str:
            """生成文件名"""
            amp_str = f"{param.amplitude:.3f}"
            phase_str = f"{param.phase:.2f}pi"
            waist_mm = param.waist * 1000
            waist_str = f"w_{waist_mm:.3f}mm"
            
            return f"LG_{amp_str},{phase_str}({param.p},{param.l})_{waist_str}"
        
        def process_all_sequential(self, progress_callback=None):
            """顺序处理所有任务"""
            total_tasks = len(self.tasks)
            self.logger.info(f"开始顺序处理 {total_tasks} 个任务")
            
            for i, task in enumerate(self.tasks):
                processed_task = self.process_single_task(task)
                
                if processed_task.status == ProcessingStatus.COMPLETED:
                    self.completed_tasks.append(processed_task)
                else:
                    self.failed_tasks.append(processed_task)
                
                if progress_callback:
                    progress_callback(i + 1, total_tasks, processed_task)
            
            self.logger.info(f"顺序处理完成: {len(self.completed_tasks)} 成功, {len(self.failed_tasks)} 失败")
        
        def process_all_parallel(self, progress_callback=None):
            """并行处理所有任务"""
            total_tasks = len(self.tasks)
            self.logger.info(f"开始并行处理 {total_tasks} 个任务 (工作进程: {self.max_workers})")
            
            executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(self.process_single_task, task): task 
                    for task in self.tasks
                }
                
                completed_count = 0
                
                # 处理完成的任务
                for future in as_completed(future_to_task):
                    completed_count += 1
                    processed_task = future.result()
                    
                    if processed_task.status == ProcessingStatus.COMPLETED:
                        self.completed_tasks.append(processed_task)
                    else:
                        self.failed_tasks.append(processed_task)
                    
                    if progress_callback:
                        progress_callback(completed_count, total_tasks, processed_task)
            
            self.logger.info(f"并行处理完成: {len(self.completed_tasks)} 成功, {len(self.failed_tasks)} 失败")
        
        def process_with_retry(self, max_retries: int = 3, progress_callback=None):
            """带重试机制的处理"""
            self.logger.info(f"开始处理任务，最大重试次数: {max_retries}")
            
            retry_tasks = self.tasks.copy()
            
            for attempt in range(max_retries + 1):
                if not retry_tasks:
                    break
                
                self.logger.info(f"第 {attempt + 1} 次尝试，剩余任务: {len(retry_tasks)}")
                
                # 处理当前轮次的任务
                current_failed = []
                
                for task in retry_tasks:
                    if attempt > 0:
                        # 重置任务状态
                        task.status = ProcessingStatus.PENDING
                        task.error_message = ""
                    
                    processed_task = self.process_single_task(task)
                    
                    if processed_task.status == ProcessingStatus.COMPLETED:
                        self.completed_tasks.append(processed_task)
                    else:
                        current_failed.append(processed_task)
                
                retry_tasks = current_failed
            
            # 记录最终失败的任务
            self.failed_tasks.extend(retry_tasks)
            
            self.logger.info(f"重试处理完成: {len(self.completed_tasks)} 成功, {len(self.failed_tasks)} 失败")
        
        def generate_processing_report(self, report_path: Path):
            """生成处理报告"""
            try:
                workbook = Workbook()
                
                # 概览页
                summary_sheet = workbook.active
                summary_sheet.title = "处理概览"
                
                total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
                success_rate = len(self.completed_tasks) / total_tasks * 100 if total_tasks > 0 else 0
                
                # 总体统计
                summary_data = [
                    ("处理概览", ""),
                    ("总任务数", total_tasks),
                    ("成功任务数", len(self.completed_tasks)),
                    ("失败任务数", len(self.failed_tasks)),
                    ("成功率", f"{success_rate:.1f}%"),
                    ("", ""),
                ]
                
                # 计算处理时间统计
                if self.completed_tasks:
                    durations = [task.duration for task in self.completed_tasks if task.duration]
                    if durations:
                        avg_duration = sum(durations) / len(durations)
                        min_duration = min(durations)
                        max_duration = max(durations)
                        total_duration = sum(durations)
                        
                        summary_data.extend([
                            ("时间统计", ""),
                            ("平均处理时间(秒)", f"{avg_duration:.2f}"),
                            ("最短处理时间(秒)", f"{min_duration:.2f}"),
                            ("最长处理时间(秒)", f"{max_duration:.2f}"),
                            ("总处理时间(秒)", f"{total_duration:.2f}"),
                        ])
                
                # 写入概览数据
                for row, (label, value) in enumerate(summary_data, 1):
                    summary_sheet.cell(row=row, column=1, value=label)
                    summary_sheet.cell(row=row, column=2, value=value)
                    
                    if label in ["处理概览", "时间统计"]:
                        summary_sheet.cell(row=row, column=1).font = Font(bold=True)
                
                # 成功任务详情
                if self.completed_tasks:
                    success_sheet = workbook.create_sheet("成功任务")
                    success_headers = ['任务ID', 'l值', 'p值', '幅度', '相位', '光腰(mm)', '处理时间(秒)', '输出文件数']
                    
                    for col, header in enumerate(success_headers, 1):
                        cell = success_sheet.cell(row=1, column=col)
                        cell.value = header
                        cell.font = Font(bold=True)
                    
                    for row, task in enumerate(self.completed_tasks, 2):
                        param = task.parameters
                        success_sheet.cell(row=row, column=1, value=task.id)
                        success_sheet.cell(row=row, column=2, value=param.l)
                        success_sheet.cell(row=row, column=3, value=param.p)
                        success_sheet.cell(row=row, column=4, value=f"{param.amplitude:.3f}")
                        success_sheet.cell(row=row, column=5, value=f"{param.phase:.3f}")
                        success_sheet.cell(row=row, column=6, value=f"{param.waist * 1000:.3f}")
                        success_sheet.cell(row=row, column=7, value=f"{task.duration:.2f}" if task.duration else "")
                        success_sheet.cell(row=row, column=8, value=len(task.result_files))
                
                # 失败任务详情
                if self.failed_tasks:
                    failed_sheet = workbook.create_sheet("失败任务")
                    failed_headers = ['任务ID', 'l值', 'p值', '幅度', '相位', '光腰(mm)', '错误信息']
                    
                    for col, header in enumerate(failed_headers, 1):
                        cell = failed_sheet.cell(row=1, column=col)
                        cell.value = header
                        cell.font = Font(bold=True)
                    
                    for row, task in enumerate(self.failed_tasks, 2):
                        param = task.parameters
                        failed_sheet.cell(row=row, column=1, value=task.id)
                        failed_sheet.cell(row=row, column=2, value=param.l)
                        failed_sheet.cell(row=row, column=3, value=param.p)
                        failed_sheet.cell(row=row, column=4, value=f"{param.amplitude:.3f}")
                        failed_sheet.cell(row=row, column=5, value=f"{param.phase:.3f}")
                        failed_sheet.cell(row=row, column=6, value=f"{param.waist * 1000:.3f}")
                        failed_sheet.cell(row=row, column=7, value=task.error_message)
                
                # 自动调整列宽
                for sheet in workbook.worksheets:
                    for column in sheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        adjusted_width = min(max_length + 2, 30)
                        sheet.column_dimensions[column_letter].width = adjusted_width
                
                # 保存报告
                report_path.parent.mkdir(parents=True, exist_ok=True)
                workbook.save(report_path)
                self.logger.info(f"处理报告已生成: {report_path}")
                
            except Exception as e:
                self.logger.error(f"生成处理报告失败: {e}")
                raise
    
    return BatchProcessor, ProcessingTask, ProcessingStatus
```

---

**文档状态**: 第10章详细内容已完成 ✅

**已完成内容**:
- pathlib模块的全面应用
- 安全的文件操作实践
- 临时文件管理
- 文件监控与自动化处理
- 二进制文件操作
- 文件名生成与验证

**下一步**: 继续完成第11章Excel处理与第12章配置管理内容

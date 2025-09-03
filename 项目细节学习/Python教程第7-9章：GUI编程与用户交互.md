# Python 教程第7-9章：GUI编程与用户交互

> **本文档涵盖**：第7章 Tkinter GUI基础与布局管理、第8章 事件驱动编程与多线程、第9章 图像显示与交互式预览

---

## 第7章：Tkinter GUI 基础与布局管理

### 7.1 Tkinter 核心组件与OAM项目应用

#### 7.1.1 窗口与容器组件详解
```python
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from typing import Dict, Any, Optional, Callable
import threading
import time

def comprehensive_tkinter_basics():
    """Tkinter基础组件全面教学"""
    
    class MainApplicationWindow:
        """主应用程序窗口 - 展示OAM项目的窗口结构"""
        
        def __init__(self):
            self.root = tk.Tk()
            self.setup_main_window()
            self.create_menu_system()
            self.create_toolbar()
            self.create_main_layout()
            self.setup_status_bar()
            
            # 存储子窗口引用
            self.child_windows = {}
            self.dialog_results = {}
        
        def setup_main_window(self):
            """设置主窗口属性"""
            # 窗口基本属性
            self.root.title("OAM 全息图生成与量子层析系统")
            self.root.geometry("1400x900")  # 宽x高
            self.root.minsize(800, 600)     # 最小尺寸
            
            # 窗口图标（如果有的话）
            try:
                self.root.iconbitmap("icon.ico")
            except tk.TclError:
                pass  # 图标文件不存在时忽略
            
            # 窗口关闭事件
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # 设置窗口样式
            self.root.configure(bg='#f0f0f0')
            
            # 窗口居中显示
            self.center_window()
        
        def center_window(self):
            """将窗口居中显示"""
            self.root.update_idletasks()
            
            # 获取窗口尺寸
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            # 获取屏幕尺寸
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # 计算居中位置
            x = (screen_width // 2) - (window_width // 2)
            y = (screen_height // 2) - (window_height // 2)
            
            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        def create_menu_system(self):
            """创建菜单系统 - 展示完整的菜单结构"""
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # 文件菜单
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="文件", menu=file_menu)
            
            file_menu.add_command(label="新建配置", command=self.new_config, accelerator="Ctrl+N")
            file_menu.add_command(label="打开配置", command=self.load_config, accelerator="Ctrl+O")
            file_menu.add_command(label="保存配置", command=self.save_config, accelerator="Ctrl+S")
            file_menu.add_command(label="另存为...", command=self.save_as_config, accelerator="Ctrl+Shift+S")
            file_menu.add_separator()
            
            # 最近文件子菜单
            recent_menu = tk.Menu(file_menu, tearoff=0)
            file_menu.add_cascade(label="最近的文件", menu=recent_menu)
            recent_menu.add_command(label="config1.json")
            recent_menu.add_command(label="config2.json")
            
            file_menu.add_separator()
            file_menu.add_command(label="退出", command=self.on_closing, accelerator="Alt+F4")
            
            # 编辑菜单
            edit_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="编辑", menu=edit_menu)
            
            edit_menu.add_command(label="撤销", command=self.undo_action, accelerator="Ctrl+Z")
            edit_menu.add_command(label="重做", command=self.redo_action, accelerator="Ctrl+Y")
            edit_menu.add_separator()
            edit_menu.add_command(label="复制参数", command=self.copy_parameters, accelerator="Ctrl+C")
            edit_menu.add_command(label="粘贴参数", command=self.paste_parameters, accelerator="Ctrl+V")
            
            # 工具菜单
            tools_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="工具", menu=tools_menu)
            
            tools_menu.add_command(label="参数验证器", command=self.open_parameter_validator)
            tools_menu.add_command(label="批量生成", command=self.open_batch_generator)
            tools_menu.add_command(label="性能监视器", command=self.open_performance_monitor)
            tools_menu.add_separator()
            tools_menu.add_command(label="首选项", command=self.open_preferences)
            
            # 帮助菜单
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="帮助", menu=help_menu)
            
            help_menu.add_command(label="用户手册", command=self.show_user_manual)
            help_menu.add_command(label="快捷键列表", command=self.show_shortcuts)
            help_menu.add_separator()
            help_menu.add_command(label="关于", command=self.show_about)
            
            # 绑定快捷键
            self.setup_keyboard_shortcuts()
        
        def setup_keyboard_shortcuts(self):
            """设置键盘快捷键"""
            shortcuts = {
                '<Control-n>': self.new_config,
                '<Control-o>': self.load_config,
                '<Control-s>': self.save_config,
                '<Control-Shift-S>': self.save_as_config,
                '<Control-z>': self.undo_action,
                '<Control-y>': self.redo_action,
                '<Control-c>': self.copy_parameters,
                '<Control-v>': self.paste_parameters,
                '<F1>': self.show_user_manual,
                '<F5>': self.refresh_display,
                '<Escape>': self.cancel_operation
            }
            
            for key, command in shortcuts.items():
                self.root.bind(key, lambda e, cmd=command: cmd())
        
        def create_toolbar(self):
            """创建工具栏"""
            toolbar_frame = ttk.Frame(self.root)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
            
            # 工具栏按钮
            buttons = [
                ("新建", "📄", self.new_config),
                ("打开", "📁", self.load_config),
                ("保存", "💾", self.save_config),
                (None, None, None),  # 分隔符
                ("生成", "⚡", self.generate_hologram),
                ("预览", "👁", self.preview_result),
                ("导出", "📤", self.export_results),
                (None, None, None),  # 分隔符
                ("设置", "⚙", self.open_preferences),
                ("帮助", "❓", self.show_user_manual)
            ]
            
            for i, (text, icon, command) in enumerate(buttons):
                if text is None:
                    # 分隔符
                    separator = ttk.Separator(toolbar_frame, orient=tk.VERTICAL)
                    separator.pack(side=tk.LEFT, fill=tk.Y, padx=5)
                else:
                    btn = ttk.Button(toolbar_frame, text=f"{icon} {text}", 
                                   command=command, width=10)
                    btn.pack(side=tk.LEFT, padx=2)
        
        def create_main_layout(self):
            """创建主要布局 - 展示复杂的布局管理"""
            # 主要内容区域
            main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 左侧面板 - 参数设置
            left_frame = ttk.LabelFrame(main_paned, text="参数设置", width=400)
            main_paned.add(left_frame, weight=1)
            
            # 右侧面板 - 预览和结果
            right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
            main_paned.add(right_paned, weight=2)
            
            # 右上 - 图像预览
            preview_frame = ttk.LabelFrame(right_paned, text="图像预览")
            right_paned.add(preview_frame, weight=2)
            
            # 右下 - 日志和控制
            control_frame = ttk.LabelFrame(right_paned, text="控制与日志")
            right_paned.add(control_frame, weight=1)
            
            # 创建各个区域的内容
            self.create_parameter_panel(left_frame)
            self.create_preview_panel(preview_frame)
            self.create_control_panel(control_frame)
        
        def create_parameter_panel(self, parent):
            """创建参数设置面板"""
            # 使用Notebook创建多标签页
            notebook = ttk.Notebook(parent)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 系统参数页
            system_frame = ttk.Frame(notebook)
            notebook.add(system_frame, text="系统参数")
            self.create_system_parameters(system_frame)
            
            # 光学参数页
            optical_frame = ttk.Frame(notebook)
            notebook.add(optical_frame, text="光学参数")
            self.create_optical_parameters(optical_frame)
            
            # 模式参数页
            mode_frame = ttk.Frame(notebook)
            notebook.add(mode_frame, text="模式参数")
            self.create_mode_parameters(mode_frame)
            
            # 输出设置页
            output_frame = ttk.Frame(notebook)
            notebook.add(output_frame, text="输出设置")
            self.create_output_settings(output_frame)
        
        def create_system_parameters(self, parent):
            """创建系统参数设置"""
            # 使用Grid布局管理器
            row = 0
            
            # 图像尺寸设置
            ttk.Label(parent, text="图像尺寸设置", font=("Arial", 10, "bold")).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
            row += 1
            
            ttk.Label(parent, text="宽度 (H):").grid(row=row, column=0, sticky=tk.W, padx=(20, 5))
            self.width_var = tk.StringVar(value="1920")
            width_spinbox = ttk.Spinbox(parent, from_=100, to=4096, textvariable=self.width_var, width=15)
            width_spinbox.grid(row=row, column=1, sticky=tk.W, padx=5)
            row += 1
            
            ttk.Label(parent, text="高度 (V):").grid(row=row, column=0, sticky=tk.W, padx=(20, 5))
            self.height_var = tk.StringVar(value="1152")
            height_spinbox = ttk.Spinbox(parent, from_=100, to=4096, textvariable=self.height_var, width=15)
            height_spinbox.grid(row=row, column=1, sticky=tk.W, padx=5)
            row += 1
            
            # 像素尺寸
            ttk.Label(parent, text="像素尺寸 (m):").grid(row=row, column=0, sticky=tk.W, padx=(20, 5))
            self.pixel_size_var = tk.StringVar(value="1.25e-5")
            pixel_entry = ttk.Entry(parent, textvariable=self.pixel_size_var, width=15)
            pixel_entry.grid(row=row, column=1, sticky=tk.W, padx=5)
            row += 1
            
            # 添加验证
            def validate_numeric(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False
            
            vcmd = (parent.register(validate_numeric), '%P')
            pixel_entry.config(validate='key', validatecommand=vcmd)
            
            # 计算模式选择
            ttk.Label(parent, text="计算模式", font=("Arial", 10, "bold")).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
            row += 1
            
            self.calc_mode_var = tk.StringVar(value="standard")
            modes = [("标准模式", "standard"), ("高精度模式", "high_precision"), ("快速模式", "fast")]
            
            for text, value in modes:
                radio = ttk.Radiobutton(parent, text=text, variable=self.calc_mode_var, value=value)
                radio.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(20, 5))
                row += 1
        
        def create_optical_parameters(self, parent):
            """创建光学参数设置"""
            # 创建滚动框架
            canvas = tk.Canvas(parent)
            scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 在滚动框架中添加内容
            row = 0
            
            # 光腰参数
            ttk.Label(scrollable_frame, text="光腰参数", font=("Arial", 10, "bold")).grid(
                row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
            row += 1
            
            ttk.Label(scrollable_frame, text="默认光腰 (m):").grid(row=row, column=0, sticky=tk.W, padx=(20, 5))
            self.waist_var = tk.DoubleVar(value=0.00254)
            waist_scale = ttk.Scale(scrollable_frame, from_=0.001, to=0.01, 
                                  variable=self.waist_var, orient=tk.HORIZONTAL, length=150)
            waist_scale.grid(row=row, column=1, sticky=tk.W, padx=5)
            
            self.waist_label = ttk.Label(scrollable_frame, text="2.54mm")
            self.waist_label.grid(row=row, column=2, sticky=tk.W, padx=5)
            
            # 绑定更新事件
            waist_scale.configure(command=self.update_waist_label)
            row += 1
            
            # 光腰修正
            self.waist_correction_var = tk.BooleanVar(value=True)
            correction_check = ttk.Checkbutton(scrollable_frame, text="启用光腰修正", 
                                             variable=self.waist_correction_var)
            correction_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(20, 5))
            row += 1
            
            # 光栅参数
            ttk.Label(scrollable_frame, text="光栅参数", font=("Arial", 10, "bold")).grid(
                row=row, column=0, columnspan=3, sticky=tk.W, pady=(20, 5))
            row += 1
            
            self.grating_enable_var = tk.BooleanVar(value=True)
            grating_check = ttk.Checkbutton(scrollable_frame, text="启用线性光栅", 
                                          variable=self.grating_enable_var,
                                          command=self.toggle_grating_params)
            grating_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(20, 5))
            row += 1
            
            # 光栅权重
            ttk.Label(scrollable_frame, text="光栅权重:").grid(row=row, column=0, sticky=tk.W, padx=(40, 5))
            self.grating_weight_var = tk.DoubleVar(value=-1.0)
            self.grating_weight_spinbox = ttk.Spinbox(scrollable_frame, from_=-10, to=10, 
                                                    increment=0.1, textvariable=self.grating_weight_var, 
                                                    width=10, format="%.1f")
            self.grating_weight_spinbox.grid(row=row, column=1, sticky=tk.W, padx=5)
            row += 1
            
            # 光栅周期
            ttk.Label(scrollable_frame, text="光栅周期:").grid(row=row, column=0, sticky=tk.W, padx=(40, 5))
            self.grating_period_var = tk.DoubleVar(value=12.0)
            self.grating_period_spinbox = ttk.Spinbox(scrollable_frame, from_=1, to=100, 
                                                    increment=0.5, textvariable=self.grating_period_var, 
                                                    width=10, format="%.1f")
            self.grating_period_spinbox.grid(row=row, column=1, sticky=tk.W, padx=5)
            row += 1
        
        def create_mode_parameters(self, parent):
            """创建模式参数设置 - 展示TreeView和动态控件"""
            # 创建TreeView来显示模式列表
            tree_frame = ttk.Frame(parent)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # TreeView
            columns = ("Index", "l", "p", "Amplitude", "Phase")
            self.mode_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=8)
            
            # 设置列标题和宽度
            for col in columns:
                self.mode_tree.heading(col, text=col)
                self.mode_tree.column(col, width=80, anchor=tk.CENTER)
            
            # 滚动条
            tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.mode_tree.yview)
            self.mode_tree.configure(yscrollcommand=tree_scrollbar.set)
            
            self.mode_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 模式编辑控件
            edit_frame = ttk.LabelFrame(parent, text="编辑模式")
            edit_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # 使用Grid布局
            ttk.Label(edit_frame, text="l值:").grid(row=0, column=0, sticky=tk.W, padx=5)
            self.l_var = tk.IntVar()
            l_spinbox = ttk.Spinbox(edit_frame, from_=-10, to=10, textvariable=self.l_var, width=8)
            l_spinbox.grid(row=0, column=1, padx=5)
            
            ttk.Label(edit_frame, text="p值:").grid(row=0, column=2, sticky=tk.W, padx=5)
            self.p_var = tk.IntVar()
            p_spinbox = ttk.Spinbox(edit_frame, from_=0, to=10, textvariable=self.p_var, width=8)
            p_spinbox.grid(row=0, column=3, padx=5)
            
            ttk.Label(edit_frame, text="幅度:").grid(row=1, column=0, sticky=tk.W, padx=5)
            self.amplitude_var = tk.DoubleVar(value=1.0)
            amplitude_entry = ttk.Entry(edit_frame, textvariable=self.amplitude_var, width=8)
            amplitude_entry.grid(row=1, column=1, padx=5)
            
            ttk.Label(edit_frame, text="相位(π):").grid(row=1, column=2, sticky=tk.W, padx=5)
            self.phase_var = tk.DoubleVar()
            phase_entry = ttk.Entry(edit_frame, textvariable=self.phase_var, width=8)
            phase_entry.grid(row=1, column=3, padx=5)
            
            # 按钮
            button_frame = ttk.Frame(edit_frame)
            button_frame.grid(row=2, column=0, columnspan=4, pady=10)
            
            ttk.Button(button_frame, text="添加模式", command=self.add_mode).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="更新模式", command=self.update_mode).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="删除模式", command=self.delete_mode).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="清空所有", command=self.clear_modes).pack(side=tk.LEFT, padx=5)
            
            # 绑定选择事件
            self.mode_tree.bind("<<TreeviewSelect>>", self.on_mode_select)
            
            # 添加一些默认模式
            self.add_default_modes()
        
        def create_output_settings(self, parent):
            """创建输出设置"""
            # 文件格式选择
            format_frame = ttk.LabelFrame(parent, text="文件格式")
            format_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.format_var = tk.StringVar(value="BMP")
            formats = ["BMP", "PNG", "TIFF", "JPEG"]
            
            for i, fmt in enumerate(formats):
                row, col = divmod(i, 2)
                ttk.Radiobutton(format_frame, text=fmt, variable=self.format_var, 
                              value=fmt).grid(row=row, column=col, sticky=tk.W, padx=20, pady=2)
            
            # 输出目录选择
            dir_frame = ttk.LabelFrame(parent, text="输出目录")
            dir_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.output_dir_var = tk.StringVar(value="./output")
            dir_entry = ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=40)
            dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
            
            ttk.Button(dir_frame, text="浏览", command=self.browse_output_dir).pack(side=tk.RIGHT, padx=5, pady=5)
            
            # 文件名前缀
            prefix_frame = ttk.LabelFrame(parent, text="文件名设置")
            prefix_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(prefix_frame, text="前缀:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            self.prefix_var = tk.StringVar(value="LG_")
            prefix_entry = ttk.Entry(prefix_frame, textvariable=self.prefix_var, width=20)
            prefix_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
            
            self.auto_timestamp_var = tk.BooleanVar()
            timestamp_check = ttk.Checkbutton(prefix_frame, text="自动添加时间戳", 
                                            variable=self.auto_timestamp_var)
            timestamp_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        def create_preview_panel(self, parent):
            """创建预览面板"""
            # 这里会在第9章详细实现
            preview_label = ttk.Label(parent, text="图像预览区域\n(将在第9章详细实现)", 
                                    justify=tk.CENTER, font=("Arial", 12))
            preview_label.pack(expand=True)
        
        def create_control_panel(self, parent):
            """创建控制面板"""
            # 创建Notebook用于不同的控制区域
            control_notebook = ttk.Notebook(parent)
            control_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 生成控制页
            generate_frame = ttk.Frame(control_notebook)
            control_notebook.add(generate_frame, text="生成控制")
            
            # 进度条
            ttk.Label(generate_frame, text="生成进度:").pack(anchor=tk.W, padx=5, pady=5)
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(generate_frame, variable=self.progress_var, 
                                              maximum=100, length=300)
            self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
            
            # 状态标签
            self.status_var = tk.StringVar(value="就绪")
            status_label = ttk.Label(generate_frame, textvariable=self.status_var)
            status_label.pack(anchor=tk.W, padx=5, pady=2)
            
            # 控制按钮
            button_frame = ttk.Frame(generate_frame)
            button_frame.pack(fill=tk.X, padx=5, pady=10)
            
            self.generate_btn = ttk.Button(button_frame, text="开始生成", 
                                         command=self.start_generation, style="Accent.TButton")
            self.generate_btn.pack(side=tk.LEFT, padx=5)
            
            self.stop_btn = ttk.Button(button_frame, text="停止", command=self.stop_generation, 
                                     state=tk.DISABLED)
            self.stop_btn.pack(side=tk.LEFT, padx=5)
            
            ttk.Button(button_frame, text="预览", command=self.preview_result).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="清除", command=self.clear_results).pack(side=tk.LEFT, padx=5)
            
            # 日志页
            log_frame = ttk.Frame(control_notebook)
            control_notebook.add(log_frame, text="日志")
            
            # 日志文本区域
            self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=50)
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 日志控制
            log_control_frame = ttk.Frame(log_frame)
            log_control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(log_control_frame, text="清除日志", 
                      command=lambda: self.log_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
            ttk.Button(log_control_frame, text="保存日志", 
                      command=self.save_log).pack(side=tk.LEFT, padx=5)
            
            self.auto_scroll_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(log_control_frame, text="自动滚动", 
                          variable=self.auto_scroll_var).pack(side=tk.RIGHT, padx=5)
        
        def setup_status_bar(self):
            """设置状态栏"""
            status_frame = ttk.Frame(self.root)
            status_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            # 状态信息
            self.status_text_var = tk.StringVar(value="就绪")
            status_label = ttk.Label(status_frame, textvariable=self.status_text_var, relief=tk.SUNKEN)
            status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
            
            # 坐标显示
            self.coord_var = tk.StringVar(value="坐标: (0, 0)")
            coord_label = ttk.Label(status_frame, textvariable=self.coord_var, relief=tk.SUNKEN, width=15)
            coord_label.pack(side=tk.RIGHT, padx=2, pady=2)
            
            # 时间显示
            self.time_var = tk.StringVar()
            time_label = ttk.Label(status_frame, textvariable=self.time_var, relief=tk.SUNKEN, width=20)
            time_label.pack(side=tk.RIGHT, padx=2, pady=2)
            
            # 更新时间
            self.update_time()
        
        # === 事件处理方法 ===
        
        def update_waist_label(self, value):
            """更新光腰标签显示"""
            waist_mm = float(value) * 1000
            self.waist_label.config(text=f"{waist_mm:.2f}mm")
        
        def toggle_grating_params(self):
            """切换光栅参数的启用状态"""
            state = tk.NORMAL if self.grating_enable_var.get() else tk.DISABLED
            self.grating_weight_spinbox.config(state=state)
            self.grating_period_spinbox.config(state=state)
        
        def on_mode_select(self, event):
            """模式选择事件处理"""
            selection = self.mode_tree.selection()
            if selection:
                item = self.mode_tree.item(selection[0])
                values = item['values']
                
                self.l_var.set(int(values[1]))
                self.p_var.set(int(values[2]))
                self.amplitude_var.set(float(values[3]))
                self.phase_var.set(float(values[4]))
        
        def add_mode(self):
            """添加新模式"""
            index = len(self.mode_tree.get_children())
            self.mode_tree.insert("", tk.END, values=(
                index, self.l_var.get(), self.p_var.get(), 
                f"{self.amplitude_var.get():.3f}", f"{self.phase_var.get():.3f}"
            ))
            self.log_message(f"添加模式: l={self.l_var.get()}, p={self.p_var.get()}")
        
        def update_mode(self):
            """更新选中的模式"""
            selection = self.mode_tree.selection()
            if selection:
                self.mode_tree.item(selection[0], values=(
                    self.mode_tree.item(selection[0])['values'][0],
                    self.l_var.get(), self.p_var.get(),
                    f"{self.amplitude_var.get():.3f}", f"{self.phase_var.get():.3f}"
                ))
                self.log_message("更新模式参数")
        
        def delete_mode(self):
            """删除选中的模式"""
            selection = self.mode_tree.selection()
            if selection:
                self.mode_tree.delete(selection)
                self.log_message("删除模式")
        
        def clear_modes(self):
            """清空所有模式"""
            if messagebox.askyesno("确认", "确定要清空所有模式吗？"):
                self.mode_tree.delete(*self.mode_tree.get_children())
                self.log_message("清空所有模式")
        
        def add_default_modes(self):
            """添加默认模式"""
            default_modes = [
                (1, 0, 1.0, 0.0),
                (-1, 0, 1.0, 0.0),
                (2, 1, 0.5, 0.5)
            ]
            
            for i, (l, p, amp, phase) in enumerate(default_modes):
                self.mode_tree.insert("", tk.END, values=(i, l, p, f"{amp:.3f}", f"{phase:.3f}"))
        
        def browse_output_dir(self):
            """浏览输出目录"""
            directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
            if directory:
                self.output_dir_var.set(directory)
                self.log_message(f"设置输出目录: {directory}")
        
        def log_message(self, message):
            """添加日志消息"""
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.log_text.insert(tk.END, log_entry)
            
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)
        
        def update_time(self):
            """更新状态栏时间显示"""
            import datetime
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_var.set(current_time)
            self.root.after(1000, self.update_time)  # 每秒更新
        
        # === 菜单命令方法 ===
        
        def new_config(self):
            """新建配置"""
            if messagebox.askyesno("新建配置", "是否创建新的配置？未保存的更改将丢失。"):
                self.reset_to_defaults()
                self.log_message("创建新配置")
        
        def load_config(self):
            """加载配置"""
            filename = filedialog.askopenfilename(
                title="选择配置文件",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            if filename:
                self.log_message(f"加载配置: {filename}")
                # 这里实现配置加载逻辑
        
        def save_config(self):
            """保存配置"""
            # 实现保存逻辑
            self.log_message("保存配置")
        
        def save_as_config(self):
            """另存为配置"""
            filename = filedialog.asksaveasfilename(
                title="保存配置文件",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            if filename:
                self.log_message(f"另存配置: {filename}")
        
        def undo_action(self):
            """撤销操作"""
            self.log_message("撤销操作")
        
        def redo_action(self):
            """重做操作"""
            self.log_message("重做操作")
        
        def copy_parameters(self):
            """复制参数"""
            self.log_message("复制参数到剪贴板")
        
        def paste_parameters(self):
            """粘贴参数"""
            self.log_message("从剪贴板粘贴参数")
        
        def open_parameter_validator(self):
            """打开参数验证器"""
            self.log_message("打开参数验证器")
        
        def open_batch_generator(self):
            """打开批量生成器"""
            self.log_message("打开批量生成器")
        
        def open_performance_monitor(self):
            """打开性能监视器"""
            self.log_message("打开性能监视器")
        
        def open_preferences(self):
            """打开首选项"""
            self.log_message("打开首选项对话框")
        
        def show_user_manual(self):
            """显示用户手册"""
            messagebox.showinfo("用户手册", "用户手册功能尚未实现")
        
        def show_shortcuts(self):
            """显示快捷键列表"""
            shortcuts_text = """
常用快捷键:
Ctrl+N - 新建配置
Ctrl+O - 打开配置
Ctrl+S - 保存配置
Ctrl+Z - 撤销
Ctrl+Y - 重做
F1 - 帮助
F5 - 刷新预览
Esc - 取消操作
            """
            messagebox.showinfo("快捷键列表", shortcuts_text)
        
        def show_about(self):
            """显示关于对话框"""
            about_text = """
OAM 全息图生成与量子层析系统
版本: 1.0.0
作者: OAM团队
            """
            messagebox.showinfo("关于", about_text)
        
        def generate_hologram(self):
            """生成全息图"""
            self.start_generation()
        
        def preview_result(self):
            """预览结果"""
            self.log_message("预览生成结果")
        
        def export_results(self):
            """导出结果"""
            self.log_message("导出结果")
        
        def start_generation(self):
            """开始生成过程"""
            self.generate_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("正在生成...")
            self.log_message("开始生成全息图")
            
            # 这里会启动后台生成线程（第8章详细实现）
            self.simulate_generation()
        
        def stop_generation(self):
            """停止生成过程"""
            self.generate_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_var.set("已停止")
            self.log_message("停止生成")
        
        def clear_results(self):
            """清除结果"""
            self.log_message("清除生成结果")
        
        def save_log(self):
            """保存日志"""
            filename = filedialog.asksaveasfilename(
                title="保存日志文件",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"保存日志: {filename}")
        
        def simulate_generation(self):
            """模拟生成过程（演示进度条）"""
            def update_progress():
                for i in range(101):
                    self.progress_var.set(i)
                    self.status_var.set(f"生成进度: {i}%")
                    self.root.update_idletasks()
                    time.sleep(0.05)  # 模拟计算时间
                
                self.generate_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.status_var.set("生成完成")
                self.log_message("全息图生成完成")
            
            # 在实际项目中，这里会启动线程
            threading.Thread(target=update_progress, daemon=True).start()
        
        def refresh_display(self):
            """刷新显示"""
            self.log_message("刷新显示")
        
        def cancel_operation(self):
            """取消当前操作"""
            self.log_message("取消操作")
        
        def reset_to_defaults(self):
            """重置为默认值"""
            self.width_var.set("1920")
            self.height_var.set("1152")
            self.pixel_size_var.set("1.25e-5")
            self.calc_mode_var.set("standard")
            self.waist_var.set(0.00254)
            self.waist_correction_var.set(True)
            self.grating_enable_var.set(True)
            self.grating_weight_var.set(-1.0)
            self.grating_period_var.set(12.0)
            self.format_var.set("BMP")
            self.output_dir_var.set("./output")
            self.prefix_var.set("LG_")
            self.auto_timestamp_var.set(False)
            
            # 清空模式列表并添加默认模式
            self.mode_tree.delete(*self.mode_tree.get_children())
            self.add_default_modes()
        
        def on_closing(self):
            """窗口关闭事件"""
            if messagebox.askokcancel("退出", "确定要退出程序吗？"):
                self.root.destroy()
        
        def run(self):
            """运行应用程序"""
            self.root.mainloop()
    
    return MainApplicationWindow

# 使用示例
def demo_tkinter_application():
    """演示Tkinter应用程序"""
    app_class = comprehensive_tkinter_basics()
    app = app_class()
    
    # 不运行主循环，只是展示如何创建
    print("Tkinter应用程序已创建，包含以下功能:")
    print("- 完整的菜单系统")
    print("- 工具栏")
    print("- 多面板布局")
    print("- 参数设置界面")
    print("- 进度监控")
    print("- 日志系统")
    print("- 状态栏")
    
    return app
```

### 7.2 布局管理器深入应用

#### 7.2.1 Grid、Pack、Place布局管理器对比与选择
```python
def layout_managers_comprehensive():
    """布局管理器完整教学"""
    
    class LayoutDemoWindow:
        """布局管理器演示窗口"""
        
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("布局管理器对比演示")
            self.root.geometry("1200x800")
            
            self.create_demo_tabs()
        
        def create_demo_tabs(self):
            """创建演示标签页"""
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Grid布局演示
            grid_frame = ttk.Frame(notebook)
            notebook.add(grid_frame, text="Grid布局管理器")
            self.demo_grid_layout(grid_frame)
            
            # Pack布局演示
            pack_frame = ttk.Frame(notebook)
            notebook.add(pack_frame, text="Pack布局管理器")
            self.demo_pack_layout(pack_frame)
            
            # Place布局演示
            place_frame = ttk.Frame(notebook)
            notebook.add(place_frame, text="Place布局管理器")
            self.demo_place_layout(place_frame)
            
            # 混合布局演示
            mixed_frame = ttk.Frame(notebook)
            notebook.add(mixed_frame, text="混合布局策略")
            self.demo_mixed_layout(mixed_frame)
            
            # 响应式布局演示
            responsive_frame = ttk.Frame(notebook)
            notebook.add(responsive_frame, text="响应式布局")
            self.demo_responsive_layout(responsive_frame)
        
        def demo_grid_layout(self, parent):
            """Grid布局管理器演示"""
            # 说明文本
            info_text = """
Grid布局管理器特点:
✓ 基于表格的二维布局
✓ 精确控制组件位置
✓ 支持跨行跨列
✓ 适合表单和复杂界面
✓ 推荐用于参数设置面板
            """
            
            info_label = ttk.Label(parent, text=info_text, justify=tk.LEFT, 
                                 font=("Arial", 10), background="#f0f8ff")
            info_label.grid(row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
            
            # 基础Grid示例
            demo_frame = ttk.LabelFrame(parent, text="基础Grid布局示例")
            demo_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
            
            # 创建一个参数输入表单
            labels = ["图像宽度:", "图像高度:", "像素尺寸:", "光腰半径:", "相位偏移:"]
            variables = []
            
            for i, label_text in enumerate(labels):
                # 标签
                label = ttk.Label(demo_frame, text=label_text)
                label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
                
                # 输入框
                var = tk.StringVar(value=f"Value {i+1}")
                variables.append(var)
                entry = ttk.Entry(demo_frame, textvariable=var, width=15)
                entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                
                # 单位标签
                units = ["pixels", "pixels", "meters", "meters", "radians"]
                unit_label = ttk.Label(demo_frame, text=units[i], foreground="gray")
                unit_label.grid(row=i, column=2, sticky="w", padx=5, pady=2)
                
                # 帮助按钮
                help_btn = ttk.Button(demo_frame, text="?", width=3)
                help_btn.grid(row=i, column=3, padx=5, pady=2)
            
            # 高级Grid特性演示
            advanced_frame = ttk.LabelFrame(parent, text="高级Grid特性")
            advanced_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
            
            # 跨列组件
            spanning_label = ttk.Label(advanced_frame, text="跨列标题 (columnspan=3)", 
                                     background="lightblue")
            spanning_label.grid(row=0, column=0, columnspan=3, sticky="ew", padx=2, pady=2)
            
            # 跨行组件
            spanning_text = tk.Text(advanced_frame, width=20, height=4)
            spanning_text.insert("1.0", "跨行文本区域\n(rowspan=3)")
            spanning_text.grid(row=1, column=0, rowspan=3, sticky="nsew", padx=2, pady=2)
            
            # 填充剩余空间的组件
            for i in range(3):
                for j in range(2):
                    btn = ttk.Button(advanced_frame, text=f"按钮({i+1},{j+1})")
                    btn.grid(row=i+1, column=j+1, sticky="ew", padx=2, pady=2)
            
            # 配置权重
            advanced_frame.columnconfigure(0, weight=1)
            advanced_frame.columnconfigure(1, weight=1)
            advanced_frame.columnconfigure(2, weight=1)
            
            # Grid选项说明
            options_frame = ttk.LabelFrame(parent, text="Grid选项说明")
            options_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
            
            options_text = """
重要参数说明:
• sticky: 组件在单元格中的对齐方式 (n, s, e, w, ne, nw, se, sw, ew, ns, nsew)
• padx, pady: 外部填充
• ipadx, ipady: 内部填充
• columnspan, rowspan: 跨列/跨行
• row, column: 位置坐标
• weight: 权重，控制空间分配
            """
            
            ttk.Label(options_frame, text=options_text, justify=tk.LEFT).pack(padx=10, pady=5)
        
        def demo_pack_layout(self, parent):
            """Pack布局管理器演示"""
            # 说明文本
            info_text = """
Pack布局管理器特点:
✓ 简单的一维布局
✓ 组件按顺序排列
✓ 适合工具栏和简单布局
✓ 支持填充和扩展
✓ 不适合复杂的表格布局
            """
            
            info_label = ttk.Label(parent, text=info_text, justify=tk.LEFT, 
                                 font=("Arial", 10), background="#f0fff0")
            info_label.pack(fill=tk.X, padx=5, pady=5)
            
            # 基础Pack示例
            basic_frame = ttk.LabelFrame(parent, text="基础Pack布局示例")
            basic_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # 顶部工具栏
            toolbar = ttk.Frame(basic_frame)
            toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            
            tools = ["新建", "打开", "保存", "复制", "粘贴"]
            for tool in tools:
                btn = ttk.Button(toolbar, text=tool, width=8)
                btn.pack(side=tk.LEFT, padx=2)
            
            # 分隔符
            ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
            
            more_tools = ["设置", "帮助"]
            for tool in more_tools:
                btn = ttk.Button(toolbar, text=tool, width=8)
                btn.pack(side=tk.LEFT, padx=2)
            
            # 右对齐按钮
            ttk.Button(toolbar, text="退出").pack(side=tk.RIGHT, padx=2)
            
            # 侧边栏布局示例
            sidebar_frame = ttk.LabelFrame(parent, text="侧边栏布局示例")
            sidebar_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 左侧边栏
            left_sidebar = ttk.Frame(sidebar_frame, width=150)
            left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
            left_sidebar.pack_propagate(False)  # 保持固定宽度
            
            ttk.Label(left_sidebar, text="导航栏", font=("Arial", 10, "bold")).pack(pady=5)
            
            nav_items = ["系统参数", "光学参数", "模式设置", "输出配置"]
            for item in nav_items:
                btn = ttk.Button(left_sidebar, text=item)
                btn.pack(fill=tk.X, padx=5, pady=2)
            
            # 主内容区域
            main_content = ttk.Frame(sidebar_frame)
            main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            ttk.Label(main_content, text="主要内容区域", 
                     font=("Arial", 12, "bold")).pack(pady=20)
            
            content_text = tk.Text(main_content, wrap=tk.WORD)
            content_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            content_text.insert("1.0", "这里是主要内容区域，使用pack(fill=tk.BOTH, expand=True)填充剩余空间。")
            
            # 底部状态栏
            status_bar = ttk.Frame(sidebar_frame)
            status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
            
            ttk.Label(status_bar, text="状态: 就绪", relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(status_bar, text="时间: 12:00", relief=tk.SUNKEN, width=15).pack(side=tk.RIGHT)
            
            # Pack选项说明
            pack_options_frame = ttk.LabelFrame(parent, text="Pack选项说明")
            pack_options_frame.pack(fill=tk.X, padx=5, pady=5)
            
            pack_options_text = """
重要参数说明:
• side: 包装方向 (TOP, BOTTOM, LEFT, RIGHT)
• fill: 填充方向 (X, Y, BOTH, NONE)
• expand: 是否扩展占用额外空间 (True/False)
• padx, pady: 外部填充
• ipadx, ipady: 内部填充
• anchor: 锚点位置 (n, s, e, w, ne, nw, se, sw, center)
            """
            
            ttk.Label(pack_options_frame, text=pack_options_text, justify=tk.LEFT).pack(padx=10, pady=5)
        
        def demo_place_layout(self, parent):
            """Place布局管理器演示"""
            # 说明文本
            info_text = """
Place布局管理器特点:
✓ 绝对和相对位置控制
✓ 像素级精确定位
✓ 适合重叠布局和特殊效果
✓ 不响应窗口大小变化
✓ 谨慎使用，易造成布局混乱
            """
            
            info_label = ttk.Label(parent, text=info_text, justify=tk.LEFT, 
                                 font=("Arial", 10), background="#fff0f5")
            info_label.place(x=10, y=10, width=600, height=100)
            
            # 绝对位置示例
            abs_frame = ttk.LabelFrame(parent, text="绝对位置示例")
            abs_frame.place(x=10, y=120, width=300, height=200)
            
            # 使用绝对坐标放置组件
            ttk.Label(abs_frame, text="绝对位置 (10, 10)").place(x=10, y=10)
            ttk.Button(abs_frame, text="按钮1").place(x=10, y=40, width=80, height=25)
            ttk.Button(abs_frame, text="按钮2").place(x=100, y=40, width=80, height=25)
            
            # 重叠组件示例
            overlap_label1 = ttk.Label(abs_frame, text="底层标签", background="lightblue")
            overlap_label1.place(x=10, y=80, width=100, height=30)
            
            overlap_label2 = ttk.Label(abs_frame, text="顶层", background="lightcoral")
            overlap_label2.place(x=30, y=90, width=60, height=20)
            
            # 相对位置示例
            rel_frame = ttk.LabelFrame(parent, text="相对位置示例")
            rel_frame.place(x=320, y=120, width=300, height=200)
            
            # 使用相对坐标 (0.0-1.0)
            ttk.Label(rel_frame, text="左上角", background="lightgreen").place(relx=0.05, rely=0.1, relwidth=0.4, relheight=0.15)
            ttk.Label(rel_frame, text="右上角", background="lightblue").place(relx=0.55, rely=0.1, relwidth=0.4, relheight=0.15)
            ttk.Label(rel_frame, text="中心", background="lightyellow").place(relx=0.3, rely=0.4, relwidth=0.4, relheight=0.2)
            ttk.Label(rel_frame, text="底部", background="lightpink").place(relx=0.05, rely=0.7, relwidth=0.9, relheight=0.15)
            
            # 动态定位示例
            dynamic_frame = ttk.LabelFrame(parent, text="动态定位示例")
            dynamic_frame.place(x=10, y=330, width=610, height=150)
            
            # 可拖拽的组件示例
            self.draggable_button = ttk.Button(dynamic_frame, text="可拖拽按钮")
            self.draggable_button.place(x=50, y=30)
            
            # 绑定拖拽事件
            self.draggable_button.bind("<Button-1>", self.start_drag)
            self.draggable_button.bind("<B1-Motion>", self.do_drag)
            
            # 位置显示标签
            self.position_label = ttk.Label(dynamic_frame, text="位置: (50, 30)")
            self.position_label.place(x=200, y=30)
            
            # 动画按钮
            ttk.Button(dynamic_frame, text="动画演示", 
                      command=self.animate_button).place(x=50, y=70)
            
            # Place选项说明
            place_options_frame = ttk.LabelFrame(parent, text="Place选项说明")
            place_options_frame.place(x=10, y=490, width=610, height=120)
            
            place_options_text = """
重要参数说明:
• x, y: 绝对坐标 (像素)
• relx, rely: 相对坐标 (0.0-1.0)
• width, height: 绝对尺寸 (像素)
• relwidth, relheight: 相对尺寸 (0.0-1.0)
• anchor: 锚点位置 (n, s, e, w, ne, nw, se, sw, center)
            """
            
            ttk.Label(place_options_frame, text=place_options_text, justify=tk.LEFT).place(x=10, y=10)
        
        def demo_mixed_layout(self, parent):
            """混合布局策略演示"""
            info_text = """
混合布局策略:
✓ 外层使用Pack管理主要区域
✓ 内层使用Grid管理详细布局
✓ 特殊效果使用Place
✓ 避免在同一容器混用布局管理器
            """
            
            info_label = ttk.Label(parent, text=info_text, justify=tk.LEFT, 
                                 font=("Arial", 10), background="#f5f5dc")
            info_label.pack(fill=tk.X, padx=5, pady=5)
            
            # 创建一个复杂的混合布局示例
            main_container = ttk.Frame(parent)
            main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 顶部工具栏 (Pack)
            toolbar_frame = ttk.Frame(main_container, relief=tk.RAISED, borderwidth=1)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
            
            ttk.Label(toolbar_frame, text="工具栏 (Pack布局)").pack(side=tk.LEFT, padx=5)
            for i in range(5):
                ttk.Button(toolbar_frame, text=f"工具{i+1}", width=8).pack(side=tk.LEFT, padx=2)
            
            # 主要内容区域
            content_container = ttk.Frame(main_container)
            content_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # 左侧参数面板 (Pack + Grid组合)
            left_panel = ttk.LabelFrame(content_container, text="参数设置 (Grid布局)")
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
            
            # 在左侧面板内使用Grid布局
            param_labels = ["参数1:", "参数2:", "参数3:", "参数4:"]
            param_vars = []
            
            for i, label_text in enumerate(param_labels):
                ttk.Label(left_panel, text=label_text).grid(row=i, column=0, sticky="w", padx=5, pady=2)
                var = tk.StringVar(value=f"值{i+1}")
                param_vars.append(var)
                ttk.Entry(left_panel, textvariable=var, width=12).grid(row=i, column=1, padx=5, pady=2)
            
            # 按钮组
            button_frame = ttk.Frame(left_panel)
            button_frame.grid(row=len(param_labels), column=0, columnspan=2, pady=10)
            
            ttk.Button(button_frame, text="应用").pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="重置").pack(side=tk.LEFT, padx=2)
            
            # 右侧显示区域 (Pack)
            right_panel = ttk.LabelFrame(content_container, text="显示区域 (Pack布局)")
            right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # 标签页容器
            notebook = ttk.Notebook(right_panel)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 图像显示页
            image_frame = ttk.Frame(notebook)
            notebook.add(image_frame, text="图像")
            
            image_canvas = tk.Canvas(image_frame, bg="white", width=300, height=200)
            image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 在canvas上绘制一些示例内容
            image_canvas.create_rectangle(50, 50, 250, 150, fill="lightblue", outline="blue")
            image_canvas.create_text(150, 100, text="图像预览区域", font=("Arial", 12))
            
            # 数据页
            data_frame = ttk.Frame(notebook)
            notebook.add(data_frame, text="数据")
            
            data_text = scrolledtext.ScrolledText(data_frame, wrap=tk.WORD)
            data_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            data_text.insert("1.0", "这里显示数据内容...\n" * 10)
            
            # 底部状态栏 (Pack)
            status_frame = ttk.Frame(main_container, relief=tk.SUNKEN, borderwidth=1)
            status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)
            
            ttk.Label(status_frame, text="状态栏 (Pack布局) - 就绪").pack(side=tk.LEFT, padx=5)
            ttk.Label(status_frame, text="内存使用: 45MB").pack(side=tk.RIGHT, padx=5)
        
        def demo_responsive_layout(self, parent):
            """响应式布局演示"""
            info_text = """
响应式布局设计:
✓ 使用权重控制空间分配
✓ 设置最小尺寸限制
✓ 绑定窗口大小变化事件
✓ 动态调整组件布局
            """
            
            info_label = ttk.Label(parent, text=info_text, justify=tk.LEFT, 
                                 font=("Arial", 10), background="#f0f8ff")
            info_label.pack(fill=tk.X, padx=5, pady=5)
            
            # 响应式Grid示例
            responsive_frame = ttk.LabelFrame(parent, text="响应式Grid布局")
            responsive_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 创建一个可缩放的Grid布局
            for i in range(3):
                for j in range(4):
                    btn = ttk.Button(responsive_frame, text=f"按钮({i},{j})")
                    btn.grid(row=i, column=j, sticky="nsew", padx=2, pady=2)
            
            # 配置权重使布局响应式
            for i in range(3):
                responsive_frame.rowconfigure(i, weight=1)
            for j in range(4):
                responsive_frame.columnconfigure(j, weight=1)
            
            # 窗口大小变化监控
            size_frame = ttk.LabelFrame(parent, text="窗口尺寸监控")
            size_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.size_label = ttk.Label(size_frame, text="窗口尺寸: ")
            self.size_label.pack(padx=10, pady=5)
            
            # 绑定窗口大小变化事件
            self.root.bind("<Configure>", self.on_window_resize)
            
            # 布局建议
            tips_frame = ttk.LabelFrame(parent, text="布局选择建议")
            tips_frame.pack(fill=tk.X, padx=5, pady=5)
            
            tips_text = """
布局管理器选择指南:
• 表单、参数设置 → 使用Grid
• 工具栏、简单排列 → 使用Pack  
• 精确定位、重叠效果 → 使用Place
• 复杂界面 → 混合使用，外层Pack，内层Grid
• 响应式设计 → Grid + 权重配置
            """
            
            ttk.Label(tips_frame, text=tips_text, justify=tk.LEFT).pack(padx=10, pady=5)
        
        # === 事件处理方法 ===
        
        def start_drag(self, event):
            """开始拖拽"""
            self.drag_start_x = event.x
            self.drag_start_y = event.y
        
        def do_drag(self, event):
            """执行拖拽"""
            # 计算新位置
            x = self.draggable_button.winfo_x() - self.drag_start_x + event.x
            y = self.draggable_button.winfo_y() - self.drag_start_y + event.y
            
            # 限制在父容器内
            max_x = self.draggable_button.master.winfo_width() - self.draggable_button.winfo_width()
            max_y = self.draggable_button.master.winfo_height() - self.draggable_button.winfo_height()
            
            x = max(0, min(x, max_x))
            y = max(0, min(y, max_y))
            
            # 更新位置
            self.draggable_button.place(x=x, y=y)
            self.position_label.config(text=f"位置: ({x}, {y})")
        
        def animate_button(self):
            """按钮动画演示"""
            def move_button():
                import math
                for i in range(100):
                    x = 50 + int(50 * math.sin(i * 0.1))
                    y = 30 + int(20 * math.cos(i * 0.1))
                    self.draggable_button.place(x=x, y=y)
                    self.position_label.config(text=f"位置: ({x}, {y})")
                    self.root.update()
                    time.sleep(0.05)
            
            # 在实际应用中应该使用线程
            move_button()
        
        def on_window_resize(self, event):
            """窗口大小变化事件处理"""
            if event.widget == self.root:
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                self.size_label.config(text=f"窗口尺寸: {width} × {height}")
        
        def run(self):
            """运行演示"""
            self.root.mainloop()
    
    return LayoutDemoWindow

# 使用示例
def demo_layout_managers():
    """演示布局管理器"""
    demo_class = layout_managers_comprehensive()
    demo = demo_class()
    
    print("布局管理器演示已创建，包含:")
    print("- Grid布局详细示例")
    print("- Pack布局应用场景")
    print("- Place布局特殊效果")
    print("- 混合布局策略")
    print("- 响应式布局设计")
    
    return demo

---

## 第8章：事件驱动编程与多线程

### 8.1 事件处理机制深入

#### 8.1.1 鼠标和键盘事件处理详解
```python
def comprehensive_event_handling():
    """全面的事件处理教学"""
    
    class EventDemoWindow:
        """事件处理演示窗口"""
        
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("事件处理完整演示")
            self.root.geometry("800x600")
            
            # 事件状态追踪
            self.mouse_position = (0, 0)
            self.key_states = {}
            self.drag_data = {"x": 0, "y": 0}
            self.selection_box = None
            
            self.create_event_demos()
            self.setup_global_bindings()
        
        def create_event_demos(self):
            """创建事件演示区域"""
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 鼠标事件演示
            mouse_frame = ttk.Frame(notebook)
            notebook.add(mouse_frame, text="鼠标事件")
            self.create_mouse_demo(mouse_frame)
            
            # 键盘事件演示
            keyboard_frame = ttk.Frame(notebook)
            notebook.add(keyboard_frame, text="键盘事件")
            self.create_keyboard_demo(keyboard_frame)
            
            # 窗口事件演示
            window_frame = ttk.Frame(notebook)
            notebook.add(window_frame, text="窗口事件")
            self.create_window_demo(window_frame)
            
            # 自定义事件演示
            custom_frame = ttk.Frame(notebook)
            notebook.add(custom_frame, text="自定义事件")
            self.create_custom_event_demo(custom_frame)
        
        def create_mouse_demo(self, parent):
            """创建鼠标事件演示"""
            # 信息显示区域
            info_frame = ttk.LabelFrame(parent, text="鼠标事件信息")
            info_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.mouse_info_var = tk.StringVar(value="移动鼠标查看事件信息")
            info_label = ttk.Label(info_frame, textvariable=self.mouse_info_var)
            info_label.pack(padx=10, pady=5)
            
            # 画布演示区域
            canvas_frame = ttk.LabelFrame(parent, text="鼠标交互画布")
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.mouse_canvas = tk.Canvas(canvas_frame, bg="white", width=400, height=300)
            self.mouse_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 绑定鼠标事件
            mouse_events = {
                "<Motion>": self.on_mouse_motion,
                "<Button-1>": self.on_left_click,
                "<Button-2>": self.on_middle_click,
                "<Button-3>": self.on_right_click,
                "<ButtonRelease-1>": self.on_left_release,
                "<Double-Button-1>": self.on_double_click,
                "<B1-Motion>": self.on_drag,
                "<MouseWheel>": self.on_mouse_wheel,
                "<Enter>": self.on_mouse_enter,
                "<Leave>": self.on_mouse_leave
            }
            
            for event, handler in mouse_events.items():
                self.mouse_canvas.bind(event, handler)
            
            # 在画布上添加一些可交互对象
            self.canvas_objects = []
            
            # 可拖拽的矩形
            rect = self.mouse_canvas.create_rectangle(50, 50, 150, 100, 
                                                    fill="lightblue", outline="blue", width=2)
            self.mouse_canvas.create_text(100, 75, text="拖拽我", font=("Arial", 10))
            self.canvas_objects.append({"id": rect, "type": "rect", "movable": True})
            
            # 可点击的圆形
            circle = self.mouse_canvas.create_oval(200, 150, 280, 230, 
                                                 fill="lightgreen", outline="green", width=2)
            self.mouse_canvas.create_text(240, 190, text="点击我", font=("Arial", 10))
            self.canvas_objects.append({"id": circle, "type": "circle", "clickable": True})
            
            # 鼠标状态显示
            status_frame = ttk.Frame(parent)
            status_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.coords_var = tk.StringVar(value="坐标: (0, 0)")
            self.button_state_var = tk.StringVar(value="按钮: 无")
            self.modifier_var = tk.StringVar(value="修饰键: 无")
            
            ttk.Label(status_frame, textvariable=self.coords_var).pack(side=tk.LEFT, padx=10)
            ttk.Label(status_frame, textvariable=self.button_state_var).pack(side=tk.LEFT, padx=10)
            ttk.Label(status_frame, textvariable=self.modifier_var).pack(side=tk.LEFT, padx=10)
        
        def create_keyboard_demo(self, parent):
            """创建键盘事件演示"""
            # 键盘输入区域
            input_frame = ttk.LabelFrame(parent, text="键盘输入测试")
            input_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.keyboard_entry = tk.Entry(input_frame, font=("Arial", 12))
            self.keyboard_entry.pack(fill=tk.X, padx=10, pady=10)
            self.keyboard_entry.focus_set()
            
            # 绑定键盘事件
            keyboard_events = {
                "<KeyPress>": self.on_key_press,
                "<KeyRelease>": self.on_key_release,
                "<Control-s>": self.on_save_shortcut,
                "<Control-o>": self.on_open_shortcut,
                "<F1>": self.on_help_key,
                "<Return>": self.on_enter_key,
                "<Escape>": self.on_escape_key
            }
            
            for event, handler in keyboard_events.items():
                self.keyboard_entry.bind(event, handler)
            
            # 键盘状态显示
            status_frame = ttk.LabelFrame(parent, text="键盘状态")
            status_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.key_info_var = tk.StringVar(value="按下任意键查看信息")
            self.key_code_var = tk.StringVar(value="键码: ")
            self.modifiers_var = tk.StringVar(value="修饰键: ")
            
            ttk.Label(status_frame, textvariable=self.key_info_var).pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(status_frame, textvariable=self.key_code_var).pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(status_frame, textvariable=self.modifiers_var).pack(anchor=tk.W, padx=10, pady=2)
            
            # 热键演示
            hotkey_frame = ttk.LabelFrame(parent, text="快捷键演示")
            hotkey_frame.pack(fill=tk.X, padx=5, pady=5)
            
            hotkey_text = """
支持的快捷键:
• Ctrl+S - 保存
• Ctrl+O - 打开
• F1 - 帮助
• Enter - 确认
• Esc - 取消
            """
            
            ttk.Label(hotkey_frame, text=hotkey_text, justify=tk.LEFT).pack(padx=10, pady=5)
            
            # 按键记录器
            recorder_frame = ttk.LabelFrame(parent, text="按键记录器")
            recorder_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.key_log = scrolledtext.ScrolledText(recorder_frame, height=8)
            self.key_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 控制按钮
            control_frame = ttk.Frame(recorder_frame)
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(control_frame, text="清除记录", 
                      command=lambda: self.key_log.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
            
            self.recording_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(control_frame, text="启用记录", 
                          variable=self.recording_var).pack(side=tk.LEFT, padx=5)
        
        def create_window_demo(self, parent):
            """创建窗口事件演示"""
            # 窗口状态信息
            info_frame = ttk.LabelFrame(parent, text="窗口状态信息")
            info_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.window_size_var = tk.StringVar()
            self.window_pos_var = tk.StringVar()
            self.window_state_var = tk.StringVar()
            
            ttk.Label(info_frame, textvariable=self.window_size_var).pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(info_frame, textvariable=self.window_pos_var).pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(info_frame, textvariable=self.window_state_var).pack(anchor=tk.W, padx=10, pady=2)
            
            # 窗口操作按钮
            control_frame = ttk.LabelFrame(parent, text="窗口控制")
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(control_frame, text="最小化", 
                      command=self.minimize_window).pack(side=tk.LEFT, padx=5, pady=5)
            ttk.Button(control_frame, text="最大化", 
                      command=self.maximize_window).pack(side=tk.LEFT, padx=5, pady=5)
            ttk.Button(control_frame, text="还原", 
                      command=self.restore_window).pack(side=tk.LEFT, padx=5, pady=5)
            ttk.Button(control_frame, text="居中", 
                      command=self.center_window).pack(side=tk.LEFT, padx=5, pady=5)
            
            # 焦点事件演示
            focus_frame = ttk.LabelFrame(parent, text="焦点事件演示")
            focus_frame.pack(fill=tk.X, padx=5, pady=5)
            
            for i in range(3):
                entry = ttk.Entry(focus_frame, width=20)
                entry.pack(side=tk.LEFT, padx=5, pady=5)
                entry.bind("<FocusIn>", lambda e, num=i: self.on_focus_in(e, num))
                entry.bind("<FocusOut>", lambda e, num=i: self.on_focus_out(e, num))
            
            # 事件日志
            log_frame = ttk.LabelFrame(parent, text="窗口事件日志")
            log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.window_log = scrolledtext.ScrolledText(log_frame, height=8)
            self.window_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def create_custom_event_demo(self, parent):
            """创建自定义事件演示"""
            # 虚拟事件演示
            virtual_frame = ttk.LabelFrame(parent, text="虚拟事件演示")
            virtual_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # 创建自定义虚拟事件
            self.root.event_add("<<DataUpdated>>", "<Control-u>")
            self.root.event_add("<<ProcessComplete>>", "<Control-p>")
            
            ttk.Button(virtual_frame, text="触发数据更新事件", 
                      command=lambda: self.root.event_generate("<<DataUpdated>>")).pack(side=tk.LEFT, padx=5, pady=5)
            ttk.Button(virtual_frame, text="触发处理完成事件", 
                      command=lambda: self.root.event_generate("<<ProcessComplete>>")).pack(side=tk.LEFT, padx=5, pady=5)
            
            # 绑定虚拟事件
            self.root.bind("<<DataUpdated>>", self.on_data_updated)
            self.root.bind("<<ProcessComplete>>", self.on_process_complete)
            
            # 定时器事件演示
            timer_frame = ttk.LabelFrame(parent, text="定时器事件演示")
            timer_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.timer_count = 0
            self.timer_id = None
            self.timer_var = tk.StringVar(value="定时器未启动")
            
            ttk.Label(timer_frame, textvariable=self.timer_var).pack(padx=10, pady=5)
            
            timer_control = ttk.Frame(timer_frame)
            timer_control.pack(padx=10, pady=5)
            
            ttk.Button(timer_control, text="启动定时器", 
                      command=self.start_timer).pack(side=tk.LEFT, padx=5)
            ttk.Button(timer_control, text="停止定时器", 
                      command=self.stop_timer).pack(side=tk.LEFT, padx=5)
            
            # 事件队列演示
            queue_frame = ttk.LabelFrame(parent, text="事件队列演示")
            queue_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.event_queue_log = scrolledtext.ScrolledText(queue_frame, height=6)
            self.event_queue_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            queue_control = ttk.Frame(queue_frame)
            queue_control.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(queue_control, text="添加延迟事件", 
                      command=self.add_delayed_event).pack(side=tk.LEFT, padx=5)
            ttk.Button(queue_control, text="清除日志", 
                      command=lambda: self.event_queue_log.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        
        def setup_global_bindings(self):
            """设置全局事件绑定"""
            # 窗口事件
            self.root.bind("<Configure>", self.on_window_configure)
            self.root.bind("<Map>", self.on_window_map)
            self.root.bind("<Unmap>", self.on_window_unmap)
            self.root.bind("<FocusIn>", self.on_window_focus_in)
            self.root.bind("<FocusOut>", self.on_window_focus_out)
            
            # 更新窗口信息
            self.update_window_info()
        
        # === 鼠标事件处理器 ===
        
        def on_mouse_motion(self, event):
            """鼠标移动事件"""
            self.mouse_position = (event.x, event.y)
            self.coords_var.set(f"坐标: ({event.x}, {event.y})")
            
            # 更新鼠标信息
            info = f"鼠标移动: x={event.x}, y={event.y}"
            if event.state & 0x0100:  # 左键按下
                info += " [左键拖拽]"
            self.mouse_info_var.set(info)
        
        def on_left_click(self, event):
            """左键点击事件"""
            self.button_state_var.set("按钮: 左键")
            self.drag_data = {"x": event.x, "y": event.y}
            
            # 检查点击的对象
            clicked_item = self.mouse_canvas.find_closest(event.x, event.y)[0]
            for obj in self.canvas_objects:
                if obj["id"] == clicked_item and obj.get("clickable"):
                    self.mouse_canvas.itemconfig(clicked_item, fill="yellow")
                    self.root.after(200, lambda: self.mouse_canvas.itemconfig(clicked_item, fill="lightgreen"))
        
        def on_right_click(self, event):
            """右键点击事件"""
            self.button_state_var.set("按钮: 右键")
            
            # 创建右键菜单
            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(label="创建矩形", command=lambda: self.create_shape(event.x, event.y, "rect"))
            context_menu.add_command(label="创建圆形", command=lambda: self.create_shape(event.x, event.y, "oval"))
            context_menu.add_separator()
            context_menu.add_command(label="清除画布", command=self.clear_canvas)
            
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
        
        def on_middle_click(self, event):
            """中键点击事件"""
            self.button_state_var.set("按钮: 中键")
        
        def on_left_release(self, event):
            """左键释放事件"""
            self.button_state_var.set("按钮: 无")
        
        def on_double_click(self, event):
            """双击事件"""
            self.mouse_info_var.set(f"双击位置: ({event.x}, {event.y})")
            # 在双击位置创建文本
            self.mouse_canvas.create_text(event.x, event.y, text="双击!", fill="red", font=("Arial", 12, "bold"))
        
        def on_drag(self, event):
            """拖拽事件"""
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            # 移动选中的对象
            clicked_item = self.mouse_canvas.find_closest(self.drag_data["x"], self.drag_data["y"])[0]
            for obj in self.canvas_objects:
                if obj["id"] == clicked_item and obj.get("movable"):
                    self.mouse_canvas.move(clicked_item, dx, dy)
                    # 同时移动文本
                    text_items = self.mouse_canvas.find_overlapping(*self.mouse_canvas.bbox(clicked_item))
                    for item in text_items:
                        if self.mouse_canvas.type(item) == "text":
                            self.mouse_canvas.move(item, dx, dy)
                    break
            
            self.drag_data = {"x": event.x, "y": event.y}
        
        def on_mouse_wheel(self, event):
            """鼠标滚轮事件"""
            direction = "上" if event.delta > 0 else "下"
            self.mouse_info_var.set(f"滚轮滚动: {direction} (delta={event.delta})")
        
        def on_mouse_enter(self, event):
            """鼠标进入事件"""
            self.mouse_canvas.config(cursor="hand2")
        
        def on_mouse_leave(self, event):
            """鼠标离开事件"""
            self.mouse_canvas.config(cursor="")
            self.coords_var.set("坐标: 鼠标离开画布")
        
        # === 键盘事件处理器 ===
        
        def on_key_press(self, event):
            """按键按下事件"""
            key_info = f"按下: {event.keysym} (字符: '{event.char}', 键码: {event.keycode})"
            self.key_info_var.set(key_info)
            self.key_code_var.set(f"键码: {event.keycode}")
            
            # 检测修饰键
            modifiers = []
            if event.state & 0x0004: modifiers.append("Ctrl")
            if event.state & 0x0008: modifiers.append("Alt")
            if event.state & 0x0001: modifiers.append("Shift")
            
            self.modifiers_var.set(f"修饰键: {', '.join(modifiers) if modifiers else '无'}")
            
            # 记录按键
            if self.recording_var.get():
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] 按下: {event.keysym}\n"
                self.key_log.insert(tk.END, log_entry)
                self.key_log.see(tk.END)
        
        def on_key_release(self, event):
            """按键释放事件"""
            if self.recording_var.get():
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] 释放: {event.keysym}\n"
                self.key_log.insert(tk.END, log_entry)
                self.key_log.see(tk.END)
        
        def on_save_shortcut(self, event):
            """Ctrl+S快捷键"""
            self.key_info_var.set("快捷键: 保存 (Ctrl+S)")
            return "break"  # 阻止默认行为
        
        def on_open_shortcut(self, event):
            """Ctrl+O快捷键"""
            self.key_info_var.set("快捷键: 打开 (Ctrl+O)")
            return "break"
        
        def on_help_key(self, event):
            """F1帮助键"""
            self.key_info_var.set("快捷键: 帮助 (F1)")
            messagebox.showinfo("帮助", "这是帮助信息")
            return "break"
        
        def on_enter_key(self, event):
            """回车键"""
            content = self.keyboard_entry.get()
            if content:
                self.key_info_var.set(f"输入内容: {content}")
                messagebox.showinfo("输入确认", f"您输入了: {content}")
        
        def on_escape_key(self, event):
            """Esc键"""
            self.keyboard_entry.delete(0, tk.END)
            self.key_info_var.set("已清空输入")
        
        # === 窗口事件处理器 ===
        
        def on_window_configure(self, event):
            """窗口配置变化事件"""
            if event.widget == self.root:
                self.update_window_info()
                self.log_window_event(f"窗口大小变化: {event.width}x{event.height}")
        
        def on_window_map(self, event):
            """窗口映射事件"""
            if event.widget == self.root:
                self.log_window_event("窗口显示")
        
        def on_window_unmap(self, event):
            """窗口取消映射事件"""
            if event.widget == self.root:
                self.log_window_event("窗口隐藏")
        
        def on_window_focus_in(self, event):
            """窗口获得焦点"""
            if event.widget == self.root:
                self.log_window_event("窗口获得焦点")
        
        def on_window_focus_out(self, event):
            """窗口失去焦点"""
            if event.widget == self.root:
                self.log_window_event("窗口失去焦点")
        
        def on_focus_in(self, event, entry_num):
            """输入框获得焦点"""
            self.log_window_event(f"输入框 {entry_num+1} 获得焦点")
            event.widget.configure(style="Focused.TEntry")
        
        def on_focus_out(self, event, entry_num):
            """输入框失去焦点"""
            self.log_window_event(f"输入框 {entry_num+1} 失去焦点")
            event.widget.configure(style="TEntry")
        
        # === 自定义事件处理器 ===
        
        def on_data_updated(self, event):
            """数据更新事件"""
            self.log_event_queue("虚拟事件: 数据已更新")
        
        def on_process_complete(self, event):
            """处理完成事件"""
            self.log_event_queue("虚拟事件: 处理已完成")
        
        # === 辅助方法 ===
        
        def update_window_info(self):
            """更新窗口信息"""
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            
            self.window_size_var.set(f"窗口大小: {width} × {height}")
            self.window_pos_var.set(f"窗口位置: ({x}, {y})")
            
            state = "正常"
            if self.root.state() == "zoomed":
                state = "最大化"
            elif self.root.state() == "iconic":
                state = "最小化"
            
            self.window_state_var.set(f"窗口状态: {state}")
        
        def minimize_window(self):
            """最小化窗口"""
            self.root.iconify()
        
        def maximize_window(self):
            """最大化窗口"""
            self.root.state('zoomed')
        
        def restore_window(self):
            """还原窗口"""
            self.root.state('normal')
        
        def center_window(self):
            """窗口居中"""
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        def create_shape(self, x, y, shape_type):
            """在指定位置创建形状"""
            if shape_type == "rect":
                shape = self.mouse_canvas.create_rectangle(x-20, y-20, x+20, y+20, 
                                                         fill="lightcoral", outline="red")
            else:  # oval
                shape = self.mouse_canvas.create_oval(x-20, y-20, x+20, y+20, 
                                                    fill="lightcyan", outline="cyan")
            
            self.canvas_objects.append({"id": shape, "type": shape_type, "movable": True})
        
        def clear_canvas(self):
            """清除画布"""
            self.mouse_canvas.delete("all")
            self.canvas_objects.clear()
        
        def start_timer(self):
            """启动定时器"""
            if self.timer_id is None:
                self.timer_count = 0
                self.timer_tick()
        
        def stop_timer(self):
            """停止定时器"""
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
                self.timer_id = None
                self.timer_var.set("定时器已停止")
        
        def timer_tick(self):
            """定时器tick"""
            self.timer_count += 1
            self.timer_var.set(f"定时器: {self.timer_count} 秒")
            self.timer_id = self.root.after(1000, self.timer_tick)
        
        def add_delayed_event(self):
            """添加延迟事件"""
            delay = 2000  # 2秒延迟
            self.log_event_queue(f"添加延迟事件 (延迟{delay}ms)")
            self.root.after(delay, lambda: self.log_event_queue("延迟事件执行!"))
        
        def log_window_event(self, message):
            """记录窗口事件"""
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            self.window_log.insert(tk.END, log_entry)
            self.window_log.see(tk.END)
        
        def log_event_queue(self, message):
            """记录事件队列日志"""
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            self.event_queue_log.insert(tk.END, log_entry)
            self.event_queue_log.see(tk.END)
        
        def run(self):
            """运行演示"""
            self.root.mainloop()
    
    return EventDemoWindow
```

---

**文档状态**: 第7章已完成，第8-9章内容较多，需要继续编写

**已完成**:
- Tkinter核心组件全面应用
- 完整的菜单系统设计
- 工具栏和状态栏实现
- 复杂的多面板布局
- 三种布局管理器详细对比
- 响应式布局设计

**下一步**: 继续完成第8章事件驱动编程与第9章图像显示内容

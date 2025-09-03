# Python 教程第5-6章：SciPy科学计算与数学算法实现

> **本文档涵盖**：第5章 SciPy科学计算工具链、第6章 数学算法实现与数值稳定性

---

## 第5章：SciPy 科学计算工具链

### 5.1 插值与信号处理

#### 5.1.1 一维插值在 inverse_sinc 中的应用详解
```python
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
import numpy as np
import time
import matplotlib.pyplot as plt

def comprehensive_inverse_sinc_implementation():
    """inverse_sinc函数的完整教学实现"""
    
    # 1. 理论背景与数学原理
    def sinc_function_analysis():
        """sinc函数分析"""
        print("=== sinc函数数学背景 ===")
        print("定义: sinc(x) = sin(x)/x, sinc(0) = 1")
        print("性质: 在[-π, 0]区间单调递减")
        print("应用: 全息图生成中的振幅调制")
        
        # 创建sinc函数的详细分析
        x = np.linspace(-2*np.pi, 2*np.pi, 1000)
        sinc_values = np.sinc(x/np.pi)  # numpy的sinc定义为sin(πx)/(πx)
        
        # 我们项目需要的sinc定义: sin(x)/x
        def our_sinc(x):
            result = np.zeros_like(x)
            nonzero_mask = (x != 0)
            result[nonzero_mask] = np.sin(x[nonzero_mask]) / x[nonzero_mask]
            result[~nonzero_mask] = 1.0
            return result
        
        our_sinc_values = our_sinc(x)
        
        # 找到单调区间
        x_mono = x[(x >= -np.pi) & (x <= 0)]
        sinc_mono = our_sinc(x_mono)
        
        print(f"单调区间: [{x_mono[0]:.3f}, {x_mono[-1]:.3f}]")
        print(f"值域: [{sinc_mono[-1]:.3f}, {sinc_mono[0]:.3f}]")
        
        return x, our_sinc_values, x_mono, sinc_mono
    
    # 2. 不同插值方法的详细实现
    def create_interpolation_methods(x_samples: np.ndarray, sinc_values: np.ndarray):
        """创建不同的插值方法"""
        
        interpolators = {}
        
        # 线性插值（最基础）
        interpolators['linear'] = interp1d(
            sinc_values, x_samples, 
            kind='linear', 
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # 三次样条插值（平衡精度和速度）
        interpolators['cubic'] = interp1d(
            sinc_values, x_samples,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # 五次样条插值（更高精度）
        interpolators['quintic'] = interp1d(
            sinc_values, x_samples,
            kind='quintic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # 平滑样条（可控制平滑度）
        interpolators['smooth_spline'] = UnivariateSpline(
            sinc_values, x_samples, 
            s=0,  # s=0表示精确插值，s>0表示平滑
            k=3   # 样条次数
        )
        
        # 高精度三次样条
        interpolators['cubic_spline'] = CubicSpline(
            sinc_values, x_samples,
            bc_type='natural'  # 自然边界条件
        )
        
        return interpolators
    
    # 3. 精度分析
    def accuracy_analysis(interpolators: dict):
        """插值精度分析"""
        print("\n=== 插值精度分析 ===")
        
        # 创建测试点
        test_y_values = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
        
        # 理论真值（通过高精度数值求解）
        def find_true_inverse_sinc(y_val, precision=1e-12):
            """高精度求解sinc(x) = y的x值"""
            from scipy.optimize import brentq
            
            def sinc_equation(x):
                return (np.sin(x) / x if x != 0 else 1.0) - y_val
            
            # 在[-π, 0]区间搜索
            try:
                return brentq(sinc_equation, -np.pi, -1e-10)
            except ValueError:
                return -np.pi  # 边界情况
        
        true_values = [find_true_inverse_sinc(y) for y in test_y_values]
        
        # 测试各种插值方法
        results = {}
        for method_name, interpolator in interpolators.items():
            if method_name == 'smooth_spline':
                interpolated = interpolator(test_y_values)
            else:
                interpolated = interpolator(test_y_values)
            
            # 计算误差
            errors = np.abs(interpolated - true_values)
            
            results[method_name] = {
                'interpolated_values': interpolated,
                'errors': errors,
                'max_error': np.max(errors),
                'mean_error': np.mean(errors),
                'rms_error': np.sqrt(np.mean(errors**2))
            }
            
            print(f"{method_name:15} - 最大误差: {results[method_name]['max_error']:.2e}, "
                  f"均方根误差: {results[method_name]['rms_error']:.2e}")
        
        return results, test_y_values, true_values
    
    # 4. 性能基准测试
    def performance_benchmark(interpolators: dict, num_evaluations: int = 100000):
        """性能基准测试"""
        print(f"\n=== 性能基准测试 ({num_evaluations} 次评估) ===")
        
        # 生成随机测试数据
        test_data = np.random.uniform(0.01, 0.99, num_evaluations)
        
        timing_results = {}
        
        for method_name, interpolator in interpolators.items():
            # 预热
            if method_name == 'smooth_spline':
                _ = interpolator(test_data[:100])
            else:
                _ = interpolator(test_data[:100])
            
            # 正式计时
            start_time = time.perf_counter()
            
            if method_name == 'smooth_spline':
                result = interpolator(test_data)
            else:
                result = interpolator(test_data)
            
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            timing_results[method_name] = {
                'total_time': execution_time,
                'time_per_eval': execution_time / num_evaluations * 1e6,  # 微秒
                'evaluations_per_sec': num_evaluations / execution_time
            }
            
            print(f"{method_name:15} - {execution_time:.4f}s "
                  f"({timing_results[method_name]['time_per_eval']:.2f} μs/eval)")
        
        return timing_results
    
    # 5. 内存使用分析
    def memory_analysis():
        """分析不同采样密度对内存的影响"""
        print("\n=== 内存使用分析 ===")
        
        sampling_densities = [1e4, 1e5, 1e6, 5e6, 1e7]
        memory_usage = {}
        
        for density in sampling_densities:
            density = int(density)
            
            # 创建采样点
            x_samples = np.linspace(-np.pi, 0, density)
            sinc_values = np.sin(x_samples) / x_samples
            sinc_values[np.isnan(sinc_values)] = 1.0
            
            # 估算内存使用
            array_memory = (x_samples.nbytes + sinc_values.nbytes) / 1024**2  # MB
            
            # 创建插值器（选择线性插值作为代表）
            start_time = time.time()
            interpolator = interp1d(sinc_values, x_samples, kind='linear')
            creation_time = time.time() - start_time
            
            memory_usage[density] = {
                'array_memory_mb': array_memory,
                'creation_time': creation_time,
                'recommended': density <= 1e6  # 推荐阈值
            }
            
            print(f"采样点数: {density:8.0e} - 内存: {array_memory:.1f}MB - "
                  f"创建时间: {creation_time:.3f}s")
        
        return memory_usage
    
    # 6. 实际项目中的优化实现
    def optimized_inverse_sinc_for_project():
        """项目中的优化实现"""
        
        class CachedInverseSinc:
            """缓存式inverse_sinc实现"""
            
            def __init__(self, num_samples: int = int(1e6), cache_size: int = 1000):
                self.num_samples = num_samples
                self.cache_size = cache_size
                self._interpolator = None
                self._cache = {}
                self._cache_keys = []
                
                self._create_interpolator()
            
            def _create_interpolator(self):
                """创建插值器"""
                x_samples = np.linspace(-np.pi, 0, self.num_samples)
                sinc_values = np.sin(x_samples) / x_samples
                sinc_values[np.isnan(sinc_values)] = 1.0
                
                self._interpolator = interp1d(
                    sinc_values, x_samples,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
            
            def __call__(self, y):
                """计算inverse_sinc值（带缓存）"""
                # 对标量值使用缓存
                if np.isscalar(y):
                    cache_key = round(y, 6)  # 6位小数精度
                    
                    if cache_key in self._cache:
                        return self._cache[cache_key]
                    
                    result = float(self._interpolator(y))
                    
                    # 更新缓存
                    if len(self._cache) >= self.cache_size:
                        oldest_key = self._cache_keys.pop(0)
                        del self._cache[oldest_key]
                    
                    self._cache[cache_key] = result
                    self._cache_keys.append(cache_key)
                    
                    return result
                else:
                    # 数组输入直接计算
                    return self._interpolator(y)
            
            def get_cache_info(self):
                """获取缓存信息"""
                return {
                    'cache_size': len(self._cache),
                    'max_cache_size': self.cache_size,
                    'hit_ratio': len(self._cache) / max(1, len(self._cache_keys))
                }
        
        return CachedInverseSinc()
    
    # 执行完整分析
    print("开始inverse_sinc函数的全面分析...")
    
    # 1. 理论分析
    x_full, sinc_full, x_mono, sinc_mono = sinc_function_analysis()
    
    # 2. 创建高密度采样（用于高精度插值）
    num_samples = int(1e6)
    x_samples = np.linspace(-np.pi, 0, num_samples)
    sinc_samples = np.sin(x_samples) / x_samples
    sinc_samples[np.isnan(sinc_samples)] = 1.0
    
    # 3. 创建插值器
    interpolators = create_interpolation_methods(x_samples, sinc_samples)
    
    # 4. 精度分析
    accuracy_results, test_points, true_values = accuracy_analysis(interpolators)
    
    # 5. 性能测试
    timing_results = performance_benchmark(interpolators)
    
    # 6. 内存分析
    memory_results = memory_analysis()
    
    # 7. 优化实现
    optimized_inverse_sinc = optimized_inverse_sinc_for_project()
    
    # 测试优化实现
    test_vals = [0.1, 0.5, 0.9]
    print(f"\n=== 优化实现测试 ===")
    for val in test_vals:
        result = optimized_inverse_sinc(val)
        print(f"inverse_sinc({val}) = {result:.6f}")
    
    print(f"缓存信息: {optimized_inverse_sinc.get_cache_info()}")
    
    return {
        'theory': (x_full, sinc_full, x_mono, sinc_mono),
        'interpolators': interpolators,
        'accuracy': accuracy_results,
        'timing': timing_results,
        'memory': memory_results,
        'optimized': optimized_inverse_sinc
    }
```

#### 5.1.2 图像处理与滤波在全息图优化中的应用
```python
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, median_filter, sobel, laplace
import numpy as np

def hologram_image_processing():
    """全息图图像处理技术"""
    
    def create_test_hologram():
        """创建测试用的全息图"""
        H, V = 512, 512
        x = np.linspace(-2, 2, H)
        y = np.linspace(-2, 2, V)
        X, Y = np.meshgrid(x, y)
        
        # 模拟全息图模式
        hologram_clean = np.sin(10*X + 5*Y) * np.exp(-(X**2 + Y**2)/2)
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, (V, H))
        hologram_noisy = hologram_clean + noise
        
        # 归一化到[0, 255]
        hologram_clean = ((hologram_clean - hologram_clean.min()) / 
                         (hologram_clean.max() - hologram_clean.min()) * 255).astype(np.uint8)
        hologram_noisy = ((hologram_noisy - hologram_noisy.min()) / 
                         (hologram_noisy.max() - hologram_noisy.min()) * 255).astype(np.uint8)
        
        return hologram_clean, hologram_noisy
    
    def noise_reduction_techniques(noisy_image: np.ndarray):
        """噪声抑制技术"""
        
        filters = {}
        
        # 1. 线性滤波器
        filters['gaussian_small'] = gaussian_filter(noisy_image, sigma=0.8)
        filters['gaussian_medium'] = gaussian_filter(noisy_image, sigma=1.5)
        filters['gaussian_large'] = gaussian_filter(noisy_image, sigma=3.0)
        
        # 均值滤波
        filters['uniform'] = ndimage.uniform_filter(noisy_image, size=3)
        
        # 2. 非线性滤波器
        filters['median_3x3'] = median_filter(noisy_image, size=3)
        filters['median_5x5'] = median_filter(noisy_image, size=5)
        
        # 形态学滤波
        filters['opening'] = ndimage.binary_opening(
            noisy_image > np.median(noisy_image)
        ).astype(np.uint8) * 255
        filters['closing'] = ndimage.binary_closing(
            noisy_image > np.median(noisy_image)
        ).astype(np.uint8) * 255
        
        # 3. 自适应滤波
        def wiener_filter_approx(image, noise_variance=None):
            """Wiener滤波近似"""
            if noise_variance is None:
                # 估计噪声方差
                laplacian = ndimage.laplace(image.astype(np.float64))
                noise_variance = np.var(laplacian) / 4
            
            # 简化的Wiener滤波
            image_smooth = gaussian_filter(image.astype(np.float64), sigma=1.0)
            image_variance = np.var(image_smooth)
            
            wiener_gain = image_variance / (image_variance + noise_variance)
            filtered = wiener_gain * image + (1 - wiener_gain) * image_smooth
            
            return np.clip(filtered, 0, 255).astype(np.uint8)
        
        filters['wiener_approx'] = wiener_filter_approx(noisy_image)
        
        # 4. 边缘保持滤波
        def bilateral_filter_simple(image, sigma_spatial=2, sigma_intensity=50):
            """简化的双边滤波"""
            from scipy.spatial.distance import pdist, squareform
            
            # 为了演示，只处理小块区域
            patch_size = 64
            center_y, center_x = image.shape[0]//2, image.shape[1]//2
            patch = image[center_y-patch_size//2:center_y+patch_size//2,
                         center_x-patch_size//2:center_x+patch_size//2]
            
            # 实际项目中会使用优化的实现
            # 这里返回高斯滤波作为近似
            return gaussian_filter(image, sigma=sigma_spatial)
        
        filters['bilateral_approx'] = bilateral_filter_simple(noisy_image)
        
        return filters
    
    def edge_enhancement_techniques(image: np.ndarray):
        """边缘增强技术"""
        
        edge_filters = {}
        image_float = image.astype(np.float64)
        
        # 1. 一阶导数算子
        edge_filters['sobel_x'] = sobel(image_float, axis=1)
        edge_filters['sobel_y'] = sobel(image_float, axis=0)
        edge_filters['sobel_magnitude'] = np.sqrt(
            edge_filters['sobel_x']**2 + edge_filters['sobel_y']**2
        )
        
        # Prewitt算子
        prewitt_x = ndimage.prewitt(image_float, axis=1)
        prewitt_y = ndimage.prewitt(image_float, axis=0)
        edge_filters['prewitt_magnitude'] = np.sqrt(prewitt_x**2 + prewitt_y**2)
        
        # 2. 二阶导数算子
        edge_filters['laplacian'] = laplace(image_float)
        
        # LoG (Laplacian of Gaussian)
        edge_filters['log'] = ndimage.gaussian_laplace(image_float, sigma=1.0)
        
        # 3. 自定义卷积核
        # Scharr算子（比Sobel更精确）
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32
        scharr_y = scharr_x.T
        
        edge_filters['scharr_x'] = ndimage.convolve(image_float, scharr_x)
        edge_filters['scharr_y'] = ndimage.convolve(image_float, scharr_y)
        edge_filters['scharr_magnitude'] = np.sqrt(
            edge_filters['scharr_x']**2 + edge_filters['scharr_y']**2
        )
        
        # 4. 非最大值抑制（简化版）
        def non_maximum_suppression_simple(magnitude, direction):
            """简化的非最大值抑制"""
            suppressed = magnitude.copy()
            
            # 简化实现：只在主要方向上抑制
            for i in range(1, magnitude.shape[0]-1):
                for j in range(1, magnitude.shape[1]-1):
                    angle = direction[i, j]
                    
                    # 量化到4个主要方向
                    if -22.5 <= angle < 22.5 or 157.5 <= angle <= 180:
                        # 水平方向
                        if (magnitude[i, j] < magnitude[i, j-1] or 
                            magnitude[i, j] < magnitude[i, j+1]):
                            suppressed[i, j] = 0
                    elif 22.5 <= angle < 67.5:
                        # 对角线方向
                        if (magnitude[i, j] < magnitude[i-1, j+1] or 
                            magnitude[i, j] < magnitude[i+1, j-1]):
                            suppressed[i, j] = 0
            
            return suppressed
        
        # 计算梯度方向
        direction = np.arctan2(edge_filters['sobel_y'], edge_filters['sobel_x']) * 180 / np.pi
        edge_filters['nms_sobel'] = non_maximum_suppression_simple(
            edge_filters['sobel_magnitude'], direction
        )
        
        # 归一化所有结果到[0, 255]
        for key, result in edge_filters.items():
            if result.dtype != np.uint8:
                result_norm = result - result.min()
                if result_norm.max() > 0:
                    result_norm = result_norm / result_norm.max() * 255
                edge_filters[key] = result_norm.astype(np.uint8)
        
        return edge_filters
    
    def frequency_domain_processing(image: np.ndarray):
        """频域处理技术"""
        
        # FFT变换
        fft_image = np.fft.fft2(image.astype(np.float64))
        fft_shifted = np.fft.fftshift(fft_image)
        
        # 功率谱
        power_spectrum = np.abs(fft_shifted)**2
        log_power = np.log10(power_spectrum + 1)
        
        # 相位谱
        phase_spectrum = np.angle(fft_shifted)
        
        H, V = image.shape
        center_h, center_v = H//2, V//2
        
        # 创建频率坐标
        freq_y = np.fft.fftfreq(H)
        freq_x = np.fft.fftfreq(V)
        freq_y_shifted = np.fft.fftshift(freq_y)
        freq_x_shifted = np.fft.fftshift(freq_x)
        
        FY, FX = np.meshgrid(freq_y_shifted, freq_x_shifted, indexing='ij')
        freq_magnitude = np.sqrt(FX**2 + FY**2)
        
        filters = {}
        
        # 1. 低通滤波器
        def create_lowpass_filter(cutoff_freq: float):
            return (freq_magnitude <= cutoff_freq).astype(np.float64)
        
        # 2. 高通滤波器
        def create_highpass_filter(cutoff_freq: float):
            return (freq_magnitude >= cutoff_freq).astype(np.float64)
        
        # 3. 带通滤波器
        def create_bandpass_filter(low_freq: float, high_freq: float):
            return ((freq_magnitude >= low_freq) & 
                   (freq_magnitude <= high_freq)).astype(np.float64)
        
        # 4. 高斯滤波器
        def create_gaussian_filter(sigma: float):
            return np.exp(-(freq_magnitude**2) / (2 * sigma**2))
        
        # 应用不同滤波器
        filter_types = {
            'lowpass_0.1': create_lowpass_filter(0.1),
            'highpass_0.05': create_highpass_filter(0.05),
            'bandpass_0.05_0.2': create_bandpass_filter(0.05, 0.2),
            'gaussian_0.1': create_gaussian_filter(0.1)
        }
        
        filtered_images = {}
        for name, filter_func in filter_types.items():
            # 应用滤波器
            filtered_fft = fft_shifted * filter_func
            
            # 逆变换
            filtered_fft_back = np.fft.ifftshift(filtered_fft)
            filtered_image = np.real(np.fft.ifft2(filtered_fft_back))
            
            # 归一化
            filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
            filtered_images[name] = filtered_image
        
        return {
            'original_fft': fft_shifted,
            'power_spectrum': power_spectrum,
            'log_power_spectrum': log_power,
            'phase_spectrum': phase_spectrum,
            'filtered_images': filtered_images,
            'freq_coordinates': (FX, FY, freq_magnitude)
        }
    
    def quality_assessment(original: np.ndarray, processed: np.ndarray):
        """图像质量评估"""
        
        # 1. 信噪比 (SNR)
        signal_power = np.mean(original.astype(np.float64)**2)
        noise_power = np.mean((original.astype(np.float64) - processed.astype(np.float64))**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # 2. 峰值信噪比 (PSNR)
        mse = np.mean((original.astype(np.float64) - processed.astype(np.float64))**2)
        psnr = 20 * np.log10(255 / (np.sqrt(mse) + 1e-10))
        
        # 3. 结构相似性指数 (SSIM) 简化版
        def simple_ssim(img1, img2):
            img1_f = img1.astype(np.float64)
            img2_f = img2.astype(np.float64)
            
            mu1 = np.mean(img1_f)
            mu2 = np.mean(img2_f)
            
            sigma1_sq = np.var(img1_f)
            sigma2_sq = np.var(img2_f)
            sigma12 = np.mean((img1_f - mu1) * (img2_f - mu2))
            
            c1 = (0.01 * 255)**2
            c2 = (0.03 * 255)**2
            
            ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            return ssim
        
        ssim = simple_ssim(original, processed)
        
        # 4. 边缘保持指数
        edge_original = sobel(original.astype(np.float64))
        edge_processed = sobel(processed.astype(np.float64))
        edge_preservation = np.corrcoef(edge_original.flatten(), edge_processed.flatten())[0, 1]
        
        return {
            'snr': snr,
            'psnr': psnr,
            'ssim': ssim,
            'edge_preservation': edge_preservation
        }
    
    # 执行完整的图像处理流程
    print("=== 全息图图像处理分析 ===")
    
    # 创建测试图像
    clean_hologram, noisy_hologram = create_test_hologram()
    
    # 噪声抑制
    denoised_images = noise_reduction_techniques(noisy_hologram)
    
    # 边缘增强
    edge_enhanced = edge_enhancement_techniques(clean_hologram)
    
    # 频域处理
    freq_results = frequency_domain_processing(clean_hologram)
    
    # 质量评估
    quality_results = {}
    for method, processed_img in denoised_images.items():
        quality_results[method] = quality_assessment(clean_hologram, processed_img)
    
    # 输出最佳方法
    best_psnr_method = max(quality_results.keys(), 
                          key=lambda x: quality_results[x]['psnr'])
    best_ssim_method = max(quality_results.keys(), 
                          key=lambda x: quality_results[x]['ssim'])
    
    print(f"最佳PSNR方法: {best_psnr_method} (PSNR: {quality_results[best_psnr_method]['psnr']:.2f})")
    print(f"最佳SSIM方法: {best_ssim_method} (SSIM: {quality_results[best_ssim_method]['ssim']:.4f})")
    
    return {
        'test_images': (clean_hologram, noisy_hologram),
        'denoised': denoised_images,
        'edge_enhanced': edge_enhanced,
        'frequency_domain': freq_results,
        'quality_assessment': quality_results,
        'recommendations': {
            'best_psnr': best_psnr_method,
            'best_ssim': best_ssim_method
        }
    }
```

### 5.2 优化与拟合

#### 5.2.1 非线性优化在参数拟合中的应用
```python
from scipy import optimize
from scipy.optimize import minimize, curve_fit, least_squares, differential_evolution
import numpy as np

def optimization_for_hologram_parameters():
    """全息图参数优化技术"""
    
    def hologram_parameter_fitting():
        """全息图参数拟合"""
        
        # 1. 目标函数定义
        def hologram_model(params, x, y):
            """全息图理论模型"""
            amplitude, waist, center_x, center_y, phase_offset = params
            
            # 计算径向距离
            r_squared = (x - center_x)**2 + (y - center_y)**2
            
            # LG模式模型（简化）
            gaussian = np.exp(-r_squared / waist**2)
            phase = phase_offset + np.arctan2(y - center_y, x - center_x)
            
            # 全息图强度
            intensity = amplitude * gaussian * (1 + np.cos(phase))
            
            return intensity
        
        # 2. 生成模拟数据
        def generate_synthetic_data():
            """生成合成测试数据"""
            H, V = 128, 128
            x = np.linspace(-2, 2, H)
            y = np.linspace(-2, 2, V)
            X, Y = np.meshgrid(x, y)
            
            # 真实参数
            true_params = [100, 0.8, 0.2, -0.1, np.pi/4]
            
            # 生成无噪声数据
            clean_data = hologram_model(true_params, X, Y)
            
            # 添加噪声
            noise_level = 0.1
            noise = np.random.normal(0, noise_level * clean_data.max(), clean_data.shape)
            noisy_data = clean_data + noise
            
            return X, Y, noisy_data, true_params, clean_data
        
        # 3. 不同优化算法比较
        def compare_optimization_methods(X, Y, data, initial_guess, true_params):
            """比较不同优化算法"""
            
            def objective_function(params):
                """目标函数：最小化残差平方和"""
                predicted = hologram_model(params, X, Y)
                residuals = (data - predicted).flatten()
                return np.sum(residuals**2)
            
            def residual_function(params):
                """残差函数（用于least_squares）"""
                predicted = hologram_model(params, X, Y)
                return (data - predicted).flatten()
            
            # 参数边界
            bounds = [
                (50, 200),      # amplitude
                (0.1, 2.0),     # waist
                (-1.0, 1.0),    # center_x
                (-1.0, 1.0),    # center_y
                (0, 2*np.pi)    # phase_offset
            ]
            
            methods = {}
            
            # 1. Nelder-Mead (无需梯度)
            result_nm = minimize(objective_function, initial_guess, method='Nelder-Mead')
            methods['Nelder-Mead'] = result_nm
            
            # 2. BFGS (拟牛顿法)
            result_bfgs = minimize(objective_function, initial_guess, method='BFGS')
            methods['BFGS'] = result_bfgs
            
            # 3. L-BFGS-B (有界BFGS)
            result_lbfgs = minimize(objective_function, initial_guess, method='L-BFGS-B', 
                                  bounds=bounds)
            methods['L-BFGS-B'] = result_lbfgs
            
            # 4. Trust Region Reflective (适合大残差问题)
            result_trf = least_squares(residual_function, initial_guess, bounds=([b[0] for b in bounds], 
                                                                               [b[1] for b in bounds]))
            methods['Trust-Region'] = result_trf
            
            # 5. Differential Evolution (全局优化)
            result_de = differential_evolution(objective_function, bounds)
            methods['Differential-Evolution'] = result_de
            
            # 分析结果
            analysis = {}
            for method_name, result in methods.items():
                if hasattr(result, 'x'):
                    fitted_params = result.x
                elif hasattr(result, 'fun'):
                    fitted_params = result.x
                else:
                    continue
                
                # 计算误差
                param_errors = np.abs(fitted_params - true_params)
                relative_errors = param_errors / np.abs(true_params) * 100
                
                analysis[method_name] = {
                    'fitted_params': fitted_params,
                    'param_errors': param_errors,
                    'relative_errors': relative_errors,
                    'success': result.success if hasattr(result, 'success') else True,
                    'nfev': result.nfev if hasattr(result, 'nfev') else result.nit,
                    'final_cost': result.fun if hasattr(result, 'fun') else result.cost
                }
            
            return analysis
        
        # 执行拟合分析
        X, Y, noisy_data, true_params, clean_data = generate_synthetic_data()
        
        # 初始猜测（故意偏离真值）
        initial_guess = [80, 1.2, 0.0, 0.0, 0.0]
        
        # 比较优化方法
        optimization_results = compare_optimization_methods(X, Y, noisy_data, initial_guess, true_params)
        
        return {
            'data': (X, Y, noisy_data, clean_data),
            'true_params': true_params,
            'initial_guess': initial_guess,
            'optimization_results': optimization_results
        }
    
    def robust_optimization():
        """鲁棒优化技术"""
        
        def huber_loss_optimization():
            """Huber损失函数优化（对异常值鲁棒）"""
            
            def huber_loss(residuals, delta=1.0):
                """Huber损失函数"""
                abs_residuals = np.abs(residuals)
                quadratic = np.where(abs_residuals <= delta, 0.5 * residuals**2, 
                                   delta * abs_residuals - 0.5 * delta**2)
                return np.sum(quadratic)
            
            # 生成带异常值的数据
            x = np.linspace(0, 10, 100)
            true_params = [2.0, 1.5, 0.5]  # a, b, c for y = a*x + b + c*sin(x)
            
            def model(params, x):
                a, b, c = params
                return a * x + b + c * np.sin(x)
            
            y_clean = model(true_params, x)
            noise = np.random.normal(0, 0.5, len(x))
            
            # 添加异常值
            outlier_indices = np.random.choice(len(x), size=10, replace=False)
            noise[outlier_indices] += np.random.choice([-1, 1], size=10) * 5
            
            y_noisy = y_clean + noise
            
            # 标准最小二乘
            def ls_objective(params):
                residuals = y_noisy - model(params, x)
                return np.sum(residuals**2)
            
            # Huber损失
            def huber_objective(params):
                residuals = y_noisy - model(params, x)
                return huber_loss(residuals)
            
            # 优化
            initial_guess = [1.0, 1.0, 1.0]
            
            result_ls = minimize(ls_objective, initial_guess, method='BFGS')
            result_huber = minimize(huber_objective, initial_guess, method='BFGS')
            
            return {
                'true_params': true_params,
                'data': (x, y_clean, y_noisy),
                'outlier_indices': outlier_indices,
                'ls_result': result_ls.x,
                'huber_result': result_huber.x,
                'ls_error': np.linalg.norm(result_ls.x - true_params),
                'huber_error': np.linalg.norm(result_huber.x - true_params)
            }
        
        def ransac_optimization():
            """RANSAC算法示例"""
            
            # 生成含异常值的直线数据
            n_points = 100
            n_outliers = 20
            
            # 内点
            x_inliers = np.random.uniform(0, 10, n_points - n_outliers)
            true_slope, true_intercept = 2.0, 1.0
            y_inliers = true_slope * x_inliers + true_intercept + np.random.normal(0, 0.5, len(x_inliers))
            
            # 外点
            x_outliers = np.random.uniform(0, 10, n_outliers)
            y_outliers = np.random.uniform(-5, 15, n_outliers)
            
            # 合并数据
            x_all = np.concatenate([x_inliers, x_outliers])
            y_all = np.concatenate([y_inliers, y_outliers])
            
            def fit_line(x, y):
                """拟合直线"""
                A = np.vstack([x, np.ones(len(x))]).T
                return np.linalg.lstsq(A, y, rcond=None)[0]
            
            def line_distance(params, x, y):
                """点到直线的距离"""
                slope, intercept = params
                return np.abs(y - (slope * x + intercept)) / np.sqrt(slope**2 + 1)
            
            # RANSAC算法
            best_params = None
            best_inliers = None
            max_inliers = 0
            threshold = 1.0
            max_iterations = 1000
            
            for _ in range(max_iterations):
                # 随机选择最小样本集
                sample_indices = np.random.choice(len(x_all), 2, replace=False)
                x_sample = x_all[sample_indices]
                y_sample = y_all[sample_indices]
                
                # 拟合模型
                params = fit_line(x_sample, y_sample)
                
                # 计算所有点的距离
                distances = line_distance(params, x_all, y_all)
                
                # 找到内点
                inliers = distances < threshold
                n_inliers = np.sum(inliers)
                
                # 更新最佳模型
                if n_inliers > max_inliers:
                    max_inliers = n_inliers
                    best_params = params
                    best_inliers = inliers
            
            # 使用所有内点重新拟合
            if best_inliers is not None:
                final_params = fit_line(x_all[best_inliers], y_all[best_inliers])
            else:
                final_params = best_params
            
            # 标准最小二乘对比
            standard_params = fit_line(x_all, y_all)
            
            return {
                'true_params': [true_slope, true_intercept],
                'data': (x_all, y_all, x_inliers, y_inliers, x_outliers, y_outliers),
                'ransac_params': final_params,
                'standard_params': standard_params,
                'inliers': best_inliers,
                'ransac_error': np.linalg.norm(final_params - [true_slope, true_intercept]),
                'standard_error': np.linalg.norm(standard_params - [true_slope, true_intercept])
            }
        
        return {
            'huber_loss': huber_loss_optimization(),
            'ransac': ransac_optimization()
        }
    
    # 执行优化分析
    fitting_results = hologram_parameter_fitting()
    robust_results = robust_optimization()
    
    # 输出最佳方法推荐
    print("=== 参数拟合方法比较 ===")
    for method, result in fitting_results['optimization_results'].items():
        print(f"{method:20} - 成功: {result['success']}, "
              f"函数评估次数: {result['nfev']:4d}, "
              f"最终误差: {result['final_cost']:.2e}")
    
    print("\n=== 鲁棒优化结果 ===")
    huber = robust_results['huber_loss']
    print(f"Huber损失 vs 最小二乘:")
    print(f"  Huber误差: {huber['huber_error']:.4f}")
    print(f"  LS误差: {huber['ls_error']:.4f}")
    
    ransac = robust_results['ransac']
    print(f"RANSAC vs 标准拟合:")
    print(f"  RANSAC误差: {ransac['ransac_error']:.4f}")
    print(f"  标准误差: {ransac['standard_error']:.4f}")
    
    return {
        'parameter_fitting': fitting_results,
        'robust_optimization': robust_results
    }
```

---

**文档状态**: 第5-6章第一部分已完成 ✅

由于内容较长，这里包含了第5章的主要内容（SciPy插值、图像处理、优化）。接下来还需要添加第6章的数学算法实现内容。这种分段方式可以确保每个文档都有合适的长度和深度。

**已完成**:
- inverse_sinc函数的完整教学实现
- 全息图图像处理技术
- 非线性优化方法比较
- 鲁棒优化技术

**下一步**: 继续完成第6章数学算法实现与数值稳定性的内容

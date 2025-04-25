import pyabf
import numpy as np
from scipy.signal import peak_widths, peak_prominences
from sklearn.cluster import KMeans

def run_kmeans(path, peak_indices, params):
    """
    使用 K-Means 对检测到的峰值进行聚类，聚类基于峰值周围信号插值后的形状。
    同时计算并返回每个聚类峰值的特征（高度、宽度、突出度）。

    :param path: ABF 文件的路径。
    :param peak_indices: 检测到的峰值（负向）的索引数组。
    :param params: 包含 K-Means 参数的字典，应包含:
                     'n_clusters': 聚类数量 K。
                     'n_init': K-Means 运行次数 (可选, 默认 10)。
                     'max_iter': K-Means 最大迭代次数 (可选, 默认 300)。
                     'random_state': 随机种子 (可选, 用于复现)。
                     'min_peak_width_samples': (新增) 峰值最小宽度（样本点数），用于过滤。默认为 5。
                     'interpolation_length': (新增) 插值后的片段长度。默认为 91。
    :return: tuple: (signal_data, filtered_peak_indices, labels, features_array)
             - signal_data: 完整的原始信号数据 (abf.sweepY)。
             - filtered_peak_indices: 通过过滤条件的峰值的索引数组。
             - labels: 与 filtered_peak_indices 对应的 K-Means 聚类标签。
             - features_array: Nx3 NumPy 数组，包含 filtered_peak_indices 对应的特征：
                               [峰高绝对值, 半高宽, 突出度]。
             返回 None, None, None, None 如果出错或没有有效峰值。
    """
    print(f"开始 K-Means 处理 ABF 文件: {path}")
    print(f"接收到 {len(peak_indices)} 个初始峰值索引。")
    print(f"K-Means 参数: {params}")

    if peak_indices is None or len(peak_indices) == 0:
        print("错误：未提供峰值索引或索引列表为空。")
        return None, None, None, None

    try:
        abf = pyabf.ABF(path)
        # 假设我们处理第一个 sweep，如果需要处理特定 sweep，需要修改这里
        if abf.sweepCount == 0:
             print(f"错误：ABF 文件 '{path}' 不包含 sweep 数据。")
             return None, None, None, None
        signal_data = abf.sweepY
        # 确保信号是 NumPy 数组
        signal_data = np.array(signal_data, dtype=float)
        print(f"成功加载信号数据，长度: {len(signal_data)}")

        # --- 1. 计算所有初始峰值的特征 ---
        # 注意：peak_widths 和 peak_prominences 寻找的是正峰，
        # 而我们检测的是负峰 (基于原代码 peak_widths(-points, ...))
        # 因此，我们在计算特征时使用 -signal_data
        negative_signal = -signal_data

        # 1.1 计算峰宽 (半高宽 FWHM) 和边界
        # rel_height=0.5 表示半高宽
        try:
             results_widths = peak_widths(negative_signal, peak_indices, rel_height=0.5)
             widths = results_widths[0]
             # width_heights = results_widths[1] # 半高宽测量处的高度
             left_ips = results_widths[2]  # 左侧插值边界点 (用于后续插值)
             right_ips = results_widths[3] # 右侧插值边界点 (用于后续插值)
             print(f"计算了 {len(widths)} 个峰的宽度。")
        except Exception as e:
             print(f"计算峰宽时出错: {e}. 检查峰值索引是否有效且在信号范围内。")
             # 如果峰值索引非常靠近边界，peak_widths 可能会失败
             # 可以在这里决定是返回错误还是尝试继续（如果只有部分失败）
             # 为了简单起见，如果特征计算失败，则整体失败
             return None, None, None, None

        # 1.2 计算峰突出度
        try:
            results_prominences = peak_prominences(negative_signal, peak_indices)
            prominences = results_prominences[0]
            # prominence_left_bases = results_prominences[1] # 突出度测量的左基线点
            # prominence_right_bases = results_prominences[2] # 突出度测量的右基线点
            print(f"计算了 {len(prominences)} 个峰的突出度。")
        except Exception as e:
            print(f"计算峰突出度时出错: {e}")
            return None, None, None, None


        # 1.3 计算峰高 (负峰的绝对值)
        peak_heights_abs = np.abs(signal_data[peak_indices])
        print(f"计算了 {len(peak_heights_abs)} 个峰的高度。")


        # --- 2. 过滤峰值 ---
        min_peak_width = params.get('min_peak_width_samples', 5) # 从参数获取最小宽度阈值
        interpolation_len = params.get('interpolation_length', 91) # 从参数获取插值长度

        valid_peak_mask = []
        segments_raw = [] # 存储用于插值的原始片段
        filtered_indices_list = [] # 存储通过过滤的原始索引
        filtered_features_list = [] # 存储通过过滤的特征 [height, width, prominence]

        print(f"开始过滤峰值，宽度阈值: {min_peak_width} 样本点...")
        for i, peak_idx in enumerate(peak_indices):
            # 使用计算出的宽度进行过滤
            current_width = widths[i]
            if current_width < min_peak_width:
                # print(f"  峰值 {peak_idx} (宽度 {current_width:.2f}) 被过滤掉 (小于 {min_peak_width})。")
                continue

            # 获取插值边界 (使用 peak_widths 返回的插值边界点)
            left_b = int(np.round(left_ips[i]))
            right_b = int(np.round(right_ips[i]))

            # 边界检查和修正
            left_b = max(0, left_b)
            right_b = min(len(signal_data), right_b) # 用 len() 作为上限

            # 确保右边界大于左边界，并且片段至少有2个点用于插值
            if right_b <= left_b + 1:
                # print(f"  峰值 {peak_idx} 的插值边界无效或太窄 ({left_b} - {right_b})，被过滤掉。")
                continue

            # 提取原始片段
            raw_segment = signal_data[left_b:right_b]

            # 记录通过过滤的峰值及其信息
            valid_peak_mask.append(i) # 记录原始索引 i
            segments_raw.append(raw_segment)
            filtered_indices_list.append(peak_idx)
            filtered_features_list.append([peak_heights_abs[i], widths[i], prominences[i]])
            # print(f"  峰值 {peak_idx} 通过过滤 (宽度 {current_width:.2f})。")


        num_filtered_peaks = len(filtered_indices_list)
        print(f"过滤完成，剩下 {num_filtered_peaks} 个有效峰值。")

        if num_filtered_peaks == 0:
            print("错误：没有峰值通过过滤条件。")
            return signal_data, np.array([]), np.array([]), np.empty((0, 3))

        # 将过滤后的列表转换为 NumPy 数组
        filtered_peak_indices_arr = np.array(filtered_indices_list)
        features_array = np.array(filtered_features_list)

        # --- 3. 对有效片段进行插值 ---
        segments_interpolated = []
        print(f"开始对 {num_filtered_peaks} 个有效片段进行插值，目标长度: {interpolation_len}...")
        for i, raw_segment in enumerate(segments_raw):
            try:
                # x 轴坐标，从 0 到 len-1
                x_old = np.arange(len(raw_segment))
                # 新的 x 轴坐标，用于插值到指定长度
                x_new = np.linspace(0, len(raw_segment) - 1, interpolation_len)
                # 执行线性插值
                interpolated = np.interp(x_new, x_old, raw_segment)
                segments_interpolated.append(interpolated)
            except Exception as e:
                 # 一般来说，如果边界检查正确，这里的插值不应失败
                 print(f"警告：对峰值索引 {filtered_indices_list[i]} 的片段进行插值时出错: {e}。将跳过此峰值。")
                 # 需要标记此峰值以便后续移除（如果需要保持数组同步）
                 # 或者，更简单的处理是，如果插值失败，则认为整个过程失败
                 return None, None, None, None # 或者返回部分结果？取决于需求

        print(f"插值完成，得到 {len(segments_interpolated)} 个长度为 {interpolation_len} 的片段。")

        if len(segments_interpolated) != num_filtered_peaks:
             # 这理论上不应该发生，除非插值中有未捕获的错误或逻辑问题
             print("错误：插值后的片段数量与过滤后的峰值数量不匹配！")
             return None, None, None, None


        # --- 4. 执行 K-Means 聚类 ---
        n_clusters = params.get('n_clusters', 3) # 获取 K 值

        # 检查有效峰值数是否足够进行聚类
        if num_filtered_peaks < n_clusters:
            print(f"警告：有效峰值数 ({num_filtered_peaks}) 少于请求的簇数 ({n_clusters})。")
            # 可以选择报错，或者将簇数减少为峰值数
            print(f"将使用 {num_filtered_peaks} 个簇进行聚类。")
            n_clusters = num_filtered_peaks
            if n_clusters == 0: # 以防万一
                 return signal_data, filtered_peak_indices_arr, np.array([]), features_array


        print(f"准备对 {len(segments_interpolated)} 个插值片段进行 K-Means 聚类 (K={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters,
                        n_init=params.get('n_init', 10),
                        max_iter=params.get('max_iter', 300),
                        random_state=params.get('random_state', None))

        # K-Means 在插值后的片段上进行拟合
        kmeans.fit(segments_interpolated)
        labels = kmeans.labels_
        print(f"K-Means 聚类完成，得到 {len(labels)} 个标签。")

        # --- 5. 返回结果 ---
        # 返回 原始信号, 过滤后的峰值索引, 聚类标签, 过滤后的特征数组
        print("成功完成 run_kmeans 处理。")
        return signal_data, filtered_peak_indices_arr, labels, features_array

    except FileNotFoundError:
        print(f"错误：ABF 文件未找到: {path}")
        return None, None, None, None
    except ImportError:
        print("错误：缺少必要的库 (pyabf, numpy, scipy, sklearn)。请确保已安装。")
        return None, None, None, None
    except Exception as e:
        print(f"处理 ABF 文件或执行 K-Means 时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
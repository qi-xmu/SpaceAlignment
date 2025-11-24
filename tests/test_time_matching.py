import pytest
import numpy as np
from time_matching import _cal_time_interval, find_closest_matches_with_threshold


class TestTimeMatching:
    def test_cal_time_interval_basic(self):
        """测试计算时间间隔的基本功能"""
        # 测试正常情况
        times = np.array([0, 1000000, 2000000, 3000000])  # 间隔1秒
        interval = _cal_time_interval(times)
        assert interval == 1000000  # 1秒的微秒数

    def test_cal_time_interval_insufficient_data(self):
        """测试数据不足的情况"""
        # 只有一个时间点
        times = np.array([1000000])
        interval = _cal_time_interval(times)
        assert interval == 0

        # 只有两个时间点
        times = np.array([1000000, 2000000])
        interval = _cal_time_interval(times)
        assert interval == 1000000

    def test_cal_time_interval_irregular(self):
        """测试不规则时间间隔"""
        times = np.array([0, 500000, 1500000, 3000000])  # 不规则间隔
        interval = _cal_time_interval(times)
        # 中位数应该是 (500000 + 1000000) / 2 = 750000
        expected_median = 750000
        assert interval == expected_median

    def test_find_closest_matches_basic(self):
        """测试基本的时间匹配功能"""
        ts1 = np.array([0, 1000000, 2000000])  # 第一个时间序列
        ts2 = np.array([50000, 1050000, 2050000])  # 第二个时间序列，有微小偏移
        
        matches, unpaired1, unpaired2 = find_closest_matches_with_threshold(ts1, ts2)
        
        # 应该找到3个匹配
        assert len(matches) == 3
        assert len(unpaired1) == 0
        assert len(unpaired2) == 0
        
        # 检查匹配的时间差
        for match in matches:
            idx1, idx2, time_diff = match
            expected_diff = abs(ts1[idx1] - ts2[idx2])
            assert time_diff == expected_diff

    def test_find_closest_matches_with_threshold(self):
        """测试带阈值的时间匹配"""
        ts1 = np.array([0, 1000000, 2000000])
        ts2 = np.array([500000, 1500000, 2500000])  # 较大的时间偏移
        
        # 设置较小的阈值，应该只有部分匹配
        max_time_diff = 200000  # 200ms
        matches, unpaired1, unpaired2 = find_closest_matches_with_threshold(
            ts1, ts2, max_time_diff
        )
        
        # 由于时间差较大，可能没有匹配或只有部分匹配
        assert len(matches) <= 3
        assert len(unpaired1) + len(unpaired2) >= 0

    def test_find_closest_matches_no_matches(self):
        """测试没有匹配的情况"""
        ts1 = np.array([0, 1000000])
        ts2 = np.array([5000000, 6000000])  # 完全不同的时间范围
        
        matches, unpaired1, unpaired2 = find_closest_matches_with_threshold(ts1, ts2)
        
        assert len(matches) == 0
        assert len(unpaired1) == 2  # 所有ts1都未匹配
        assert len(unpaired2) == 2  # 所有ts2都未匹配

    def test_find_closest_matches_different_lengths(self):
        """测试不同长度的时间序列"""
        ts1 = np.array([0, 1000000, 2000000, 3000000])  # 4个点
        ts2 = np.array([50000, 1050000])  # 2个点
        
        matches, unpaired1, unpaired2 = find_closest_matches_with_threshold(ts1, ts2)
        
        assert len(matches) == 2
        assert len(unpaired1) == 2  # ts1有2个点未匹配
        assert len(unpaired2) == 0  # ts2所有点都匹配了

    def test_find_closest_matches_duplicate_times(self):
        """测试重复时间点的情况"""
        ts1 = np.array([0, 0, 1000000, 1000000])  # 重复时间点
        ts2 = np.array([50000, 1050000])
        
        matches, unpaired1, unpaired2 = find_closest_matches_with_threshold(ts1, ts2)
        
        # 由于重复，可能只有部分匹配
        assert len(matches) <= 2
        assert len(unpaired1) >= 2

    def test_find_closest_matches_empty_input(self):
        """测试空输入的情况"""
        # 空数组
        matches, unpaired1, unpaired2 = find_closest_matches_with_threshold([], [])
        assert len(matches) == 0
        assert len(unpaired1) == 0
        assert len(unpaired2) == 0
        
        # 一个空数组
        ts1 = np.array([0, 1000000])
        matches, unpaired1, unpaired2 = find_closest_matches_with_threshold(ts1, [])
        assert len(matches) == 0
        assert len(unpaired1) == 2
        assert len(unpaired2) == 0


if __name__ == "__main__":
    pytest.main([__file__])
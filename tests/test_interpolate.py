import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from base.interpolate import get_time_series, interpolate_vector3d, slerp_rotation


class TestSlerpRotation:
    def test_slerp_rotation_basic(self):
        """测试基本的旋转插值"""
        # 创建测试数据
        t_old_us = np.array([0, 1000000, 2000000])  # 0s, 1s, 2s
        angles = np.array([[0, 0, 0], [0, np.pi / 4, 0], [0, np.pi / 2, 0]])
        rots = Rotation.from_euler("xyz", angles)

        # 在中间时间点插值
        t_new_us = np.array([500000])  # 0.5s
        result = slerp_rotation(rots, t_old_us, t_new_us)

        # 检查结果类型和形状
        assert isinstance(result, Rotation)
        assert len(result) == len(t_new_us)

    def test_slerp_rotation_length_mismatch(self):
        """测试旋转和时间长度不匹配的情况"""
        t_old_us = np.array([0, 1000000])
        angles = np.array([[0, 0, 0], [0, np.pi / 4, 0], [0, np.pi / 2, 0]])
        rots = Rotation.from_euler("xyz", angles)

        with pytest.raises(AssertionError):
            slerp_rotation(rots, t_old_us, np.array([500000]))


class TestInterpolateVector3d:
    def test_interpolate_vector3d_basic(self):
        """测试基本的3D向量插值"""
        # 创建测试数据
        t_old_us = np.array([0, 1000000, 2000000])
        vec3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        # 在中间时间点插值
        t_new_us = np.array([500000, 1500000])
        result = interpolate_vector3d(vec3d, t_old_us, t_new_us)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)  # 2个时间点，每个点3个坐标
        assert np.allclose(result[0], [0.5, 0.5, 0.5], atol=0.1)

    def test_interpolate_vector3d_out_of_range(self):
        """测试超出时间范围的插值"""
        t_old_us = np.array([0, 1000000, 2000000])
        vec3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        # 超出范围的时间点
        t_new_us = np.array([-100000, 2500000])

        with pytest.raises(AssertionError):
            interpolate_vector3d(vec3d, t_old_us, t_new_us)

    def test_interpolate_vector3d_length_mismatch(self):
        """测试向量和时间长度不匹配的情况"""
        t_old_us = np.array([0, 1000000])
        vec3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        with pytest.raises(AssertionError):
            interpolate_vector3d(vec3d, t_old_us, np.array([500000]))


class TestGetTimeSeries:
    def test_get_time_series_basic(self):
        """测试基本的时间序列生成"""
        # 创建测试时间序列
        ts_us = [
            np.array([0, 1000000, 2000000]),  # 第一个传感器
            np.array([500000, 1500000, 2500000]),  # 第二个传感器
        ]

        result = get_time_series(ts_us, rate=100.0)  # 100Hz

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        # 应该在重叠的时间范围内生成时间序列
        assert result[0] >= 500000  # 开始时间应该是两个序列的最大开始时间
        assert result[-1] <= 2000000  # 结束时间应该是两个序列的最小结束时间

    def test_get_time_series_with_time_limits(self):
        """测试带时间限制的时间序列生成"""
        ts_us = [np.array([0, 1000000, 2000000]), np.array([500000, 1500000, 2500000])]

        # 限制时间范围：从第1秒到第1.5秒
        result = get_time_series(ts_us, t_start_s=1, t_end_s=1.5, rate=100.0)

        assert len(result) > 0
        assert result[0] >= 1000000  # 1秒
        assert result[-1] <= 1500000  # 1.5秒

    def test_get_time_series_no_overlap(self):
        """测试没有时间重叠的情况"""
        ts_us = [
            np.array([0, 1000000]),  # 0-1秒
            np.array([2000000, 3000000]),  # 2-3秒，没有重叠
        ]

        with pytest.raises(AssertionError):
            get_time_series(ts_us, rate=100.0)

    def test_get_time_series_empty_input(self):
        """测试空输入的情况"""
        with pytest.raises(ValueError):
            get_time_series([], rate=100.0)


if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from base.basetype import Transform, PoseSeries, DataCheck


class TestTransform:
    def test_transform_initialization(self):
        """测试Transform数据类的初始化"""
        rot = Rotation.identity()
        tran = np.array([1.0, 2.0, 3.0])
        transform = Transform(rot=rot, tran=tran)
        
        assert transform.rot == rot
        assert np.array_equal(transform.tran, tran)

    def test_transform_multiplication(self):
        """测试Transform的乘法运算"""
        rot1 = Rotation.from_euler('xyz', [0, 0, 0])
        tran1 = np.array([1.0, 0.0, 0.0])
        transform1 = Transform(rot1, tran1)
        
        rot2 = Rotation.from_euler('xyz', [0, 0, 0])
        tran2 = np.array([0.0, 1.0, 0.0])
        transform2 = Transform(rot2, tran2)
        
        result = transform1 * transform2
        expected_tran = np.array([1.0, 1.0, 0.0])
        
        assert np.allclose(result.tran, expected_tran)

    def test_transform_identity(self):
        """测试Transform的identity方法"""
        identity = Transform.identity()
        # Rotation对象不能直接用==比较，需要比较矩阵表示
        assert np.allclose(identity.rot.as_matrix(), Rotation.identity().as_matrix())
        assert np.array_equal(identity.tran, np.zeros(3))

    def test_transform_inverse(self):
        """测试Transform的逆变换"""
        rot = Rotation.from_euler('xyz', [np.pi/4, 0, 0])
        tran = np.array([1.0, 2.0, 3.0])
        transform = Transform(rot, tran)
        
        inverse_transform = transform.inverse()
        # 应用变换和逆变换应该得到恒等变换
        result = transform * inverse_transform
        
        assert np.allclose(result.tran, np.zeros(3), atol=1e-10)


class TestPoseSeries:
    def test_pose_series_initialization(self):
        """测试PoseSeries数据类的初始化"""
        rots = Rotation.from_euler('xyz', [[0, 0, 0], [np.pi/2, 0, 0]])
        trans = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pose_series = PoseSeries(rots=rots, trans=trans)
        
        assert len(pose_series) == 2
        assert pose_series.rots == rots
        assert np.array_equal(pose_series.trans, trans)

    def test_pose_series_inverse(self):
        """测试PoseSeries的逆变换"""
        rots = Rotation.from_euler('xyz', [[0, 0, 0], [np.pi/4, 0, 0]])
        trans = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pose_series = PoseSeries(rots=rots, trans=trans)
        
        inverse_series = pose_series.inverse()
        assert len(inverse_series) == len(pose_series)

    def test_pose_series_reset_trans(self):
        """测试重置平移向量"""
        rots = Rotation.from_euler('xyz', [[0, 0, 0], [np.pi/2, 0, 0]])
        trans = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pose_series = PoseSeries(rots=rots, trans=trans)
        
        pose_series.reset_trans()
        assert np.array_equal(pose_series.trans, np.zeros((2, 3)))


class TestDataCheck:
    def test_data_check_initialization(self):
        """测试DataCheck的初始化"""
        t_gi_us = 1000
        data_check = DataCheck(t_gi_us=t_gi_us)
        assert data_check.t_gi_us == t_gi_us

    def test_data_check_from_json(self, tmp_path):
        """测试DataCheck从JSON文件加载"""
        # 创建测试JSON文件
        test_data = {
            "check_time_diff": {
                "time_diff_21_us": 12345
            }
        }
        
        json_file = tmp_path / "test_data.json"
        import json
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        # 测试从JSON加载
        data_check = DataCheck.from_json(json_file)
        assert data_check.t_gi_us == 12345

    def test_data_check_from_json_invalid_format(self, tmp_path):
        """测试无效JSON格式的DataCheck加载"""
        invalid_data = {"invalid_key": "invalid_value"}
        
        json_file = tmp_path / "invalid_data.json"
        import json
        with open(json_file, 'w') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(AssertionError):
            DataCheck.from_json(json_file)

    def test_data_check_from_json_file_not_found(self):
        """测试从不存在文件加载DataCheck"""
        with pytest.raises(FileNotFoundError):
            DataCheck.from_json(Path("nonexistent.json"))


if __name__ == "__main__":
    pytest.main([__file__])
"""
测试 random_forest_pipeline 的核心功能
"""
import sys
import os

def test_data_loading():
    """测试 WDBC 数据集是否能正常加载"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'wdbc.data')
    assert os.path.exists(data_path), f"数据集文件不存在: {data_path}"
    with open(data_path, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 569, f"样本数应为 569，实际为 {len(lines)}"
    print("✓ test_data_loading passed")


def test_pipeline_import():
    """测试 pipeline 脚本能否正常导入"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    try:
        import random_forest_pipeline
        print("✓ test_pipeline_import passed")
    except ImportError as e:
        print(f"✗ test_pipeline_import failed: {e}")
        assert False


def test_requirements_installable():
    """测试 requirements.txt 中的包是否可以解析（不实际安装）"""
    req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    assert os.path.exists(req_path), "requirements.txt 不存在"
    with open(req_path, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    assert len(packages) > 0, "requirements.txt 为空"
    print(f"✓ test_requirements_installable passed ({len(packages)} packages)")


if __name__ == '__main__':
    test_data_loading()
    test_pipeline_import()
    test_requirements_installable()
    print("\n所有测试通过！")

"""
蔗循智策 - 测试运行器

运行 test.py 中的所有测试，并输出结构化报告。

用法：
    python run_tests.py
"""

import os
import subprocess
import sys
from datetime import datetime


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_unit_tests() -> bool:
    """运行 python test.py 并返回是否通过"""
    print("=" * 70)
    print("运行单元测试: python test.py")
    print("=" * 70)
    result = subprocess.run(
        [sys.executable, "test.py"],
        cwd=PROJECT_DIR,
        text=True,
        encoding="utf-8",
    )
    return result.returncode == 0


def main():
    start = datetime.now()
    tests_passed = run_unit_tests()

    elapsed = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 70)
    print("测试运行报告")
    print("=" * 70)
    print(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时:   {elapsed:.2f} 秒")
    print(f"单元测试: {'通过' if tests_passed else '失败'}")
    print("=" * 70)

    sys.exit(0 if tests_passed else 1)


if __name__ == "__main__":
    main()

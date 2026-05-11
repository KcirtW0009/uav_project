import os
from datetime import datetime

print("=" * 70)
print("查找 experiment_results 目录下所有文件的修改时间")
print("=" * 70)

result_dir = 'experiment_results'
files_info = []

for root, dirs, files in os.walk(result_dir):
    for file in files:
        if file.endswith('.json') or file.endswith('.pkl'):
            full_path = os.path.join(root, file)
            mtime = os.path.getmtime(full_path)
            size = os.path.getsize(full_path) / 1024  # KB
            mtime_str = datetime.fromtimestamp(mtime)
            files_info.append({
                'path': full_path,
                'mtime': mtime,
                'mtime_str': mtime_str,
                'size': size
            })

# 按修改时间排序（最新的在前）
files_info.sort(key=lambda x: x['mtime'], reverse=True)

print(f"\n共找到 {len(files_info)} 个数据文件\n")
print("-" * 80)
print(f"{'文件名':<45} {'修改时间':<22} {'大小(KB)':<10}")
print("-" * 80)

for info in files_info[:20]:  # 显示最新的20个
    path = info['path']
    name = os.path.basename(path)
    print(f"{name:<45} {info['mtime_str']:<22} {info['size']:>6.1f}")

print("\n" + "=" * 70)
print("[重点] 检查是否有今天的日期 (2026-05-11)")
print("=" * 70)

today_files = [f for f in files_info if f['mtime_str'].strftime('%Y-%m-%d') == '2026-05-11']
if today_files:
    print(f"\n找到 {len(today_files)} 个今天修改的文件:")
    for f in today_files:
        print(f"  {os.path.basename(f['path'])} - {f['mtime_str']} - {f['size']:.1f}KB")
else:
    print("\n[WARN] 没有找到今天(2026-05-11)修改的JSON/PKL文件!")
    print("       这说明最新运行的数据可能没有成功保存到磁盘")

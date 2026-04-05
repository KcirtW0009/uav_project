#!/usr/bin/env python3
# Fix: Add debug output to train() method
filepath = r'f:\桌面\本科毕业论文\结题\uav_project\uav_system\mappo_agent.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the target code
target = '        if self.buffer.ptr == 0:\n            return {}'
replacement = '''        if self.buffer.ptr == 0:
            if self._current_train_step % 100 == 0:
                print(f"[DEBUG-TRAIN] buffer.ptr=0, skip PPO update (step={self._current_train_step})")
            return {}'''

if target in content:
    content = content.replace(target, replacement)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print('[OK] Added debug output to train()')
else:
    print('[WARN] Target code not found in train() method')

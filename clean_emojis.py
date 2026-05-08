#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彻底清理finetune_multi_scenario.py中的所有Emoji字符
"""
import re

# Emoji替换映射（完整版）
EMOJI_MAP = {
    # 常用Emoji
    '📋': '[PLAN]',
    '📝': '[NOTE]',
    '💾': '[SAVE]',
    '⚡': '[FAST]',
    '🎯': '[TARGET]',
    '🔍': '[SEARCH]',
    '🚀': '[*]',
    '✅': '[OK]',
    '❌': '[FAIL]',
    '⚠️': '[WARN]',
    '🎉': '[PARTY]',
    '🔄': '[LOOP]',
    '⏳': '[WAIT]',
    '📊': '[CHART]',
    '📈': '[UP]',
    '📦': '[BOX]',
    '🏙️': '[CITY]',
    '🌾': '[FARM]',
    '🚑': '[RESCUE]',
    '🔧': '[TOOL]',
    # 其他可能遗漏的
    '🏆': '[TROPHY]',
    '🎓': '[GRAD]',
    '💡': '[IDEA]',
    '🔥': '[FIRE]',
    '⭐': '[STAR]',
    '❗': '[!]',
    '❓': '[?]',
    '📍': '[LOC]',
    '🎨': '[ART]',
    '🛠️': '[TOOLS]',
    '📁': '[FOLDER]',
    '📂': '[OPEN]',
    '🔒': '[LOCK]',
    '🔓': '[UNLOCK]',
    '🎪': '[CIRCUS]',
    '🎭': '[MASK]',
    '🏃': '[RUN]',
    '🚶': '[WALK]',
    '🏠': '[HOME]',
    '🏢': '[OFFICE]',
    '🏥': '[HOSPITAL]',
    '🏫': '[SCHOOL]',
    '🎮': '[GAME]',
    '🎬': '[MOVIE]',
    '🎵': '[MUSIC]',
    '🎤': '[MIC]',
    '🎧': '[HEADPHONE]',
    '📱': '[PHONE]',
    '💻': '[LAPTOP]',
    '🖥️': '[MONITOR]',
    '⌨️': '[KEYBOARD]',
    '🖱️': '[MOUSE]',
    '🖨️': '[PRINTER]',
    '📷': '[CAMERA]',
    '📹': '[VIDEO]',
    '🎥': '[FILM]',
    '📺': '[TV]',
    '📻': '[RADIO]',
    '⏰': '[CLOCK]',
    '⌛': '[HOURGLASS]',
    '📅': '[CALENDAR]',
    '📆': '[CALENDAR2]',
    '🌟': '[STAR2]',
    '✨': '[SPARKLE]',
    '💫': '[DIZZY]',
    '🎯': '[BULLSEYE]',
    '🏁': '[FINISH]',
    '🚩': '[FLAG]',
    '🎌': '[FLAGS]',
    '🏳️': '[WHITE_FLAG]',
    '🏴': '[BLACK_FLAG]',
    # 遗漏的Emoji (第二批)
    '📌': '[PIN]',
    '📑': '[TABS]',
    '🔴': '[RED_CIRCLE]',
    '🎲': '[DICE]',
    '🔑': '[KEY]',
}

def clean_file():
    with open('finetune_multi_scenario.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_length = len(content)
    
    # 统计替换次数
    total_replacements = 0
    for emoji, replacement in EMOJI_MAP.items():
        count = content.count(emoji)
        if count > 0:
            content = content.replace(emoji, replacement)
            total_replacements += count
            print(f'  Replaced {count}x: U+{ord(emoji):04X} -> {replacement}')
    
    with open('finetune_multi_scenario.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f'\n[OK] Cleanup completed!')
    print(f'   Total replacements: {total_replacements}')
    print(f'   File size: {original_length} -> {len(content)} bytes')
    
    # 验证：检查是否还有剩余的Emoji
    remaining = re.findall(r'[\U0001F300-\U0001F9FF\U0001FA00-\U0001FA6F]', content)
    if remaining:
        print(f'\n[WARN] Still found {len(remaining)} potential emojis!')
        unique_remaining = list(set(remaining))
        for char in unique_remaining[:10]:
            line_num = content[:content.index(char)].count('\n') + 1
            print(f'  U+{ord(char):04X} at line {line_num}')
    else:
        print(f'\n[OK] No remaining emojis detected!')

if __name__ == '__main__':
    clean_file()

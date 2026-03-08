"""Fix separator hyphens in all book markdown files."""
import os
import re

def fix_hyphens(content):
    lines = content.split('\n')
    result = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
        if in_code_block:
            result.append(line)
            continue
        # Replace double-spaced '  -  ' separator with ', '
        line = re.sub(r'  -  ', ', ', line)
        result.append(line)
    return '\n'.join(result)

changed = []
for root, dirs, files in os.walk('book'):
    for fname in files:
        if fname.endswith('.md'):
            path = os.path.join(root, fname)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_content = fix_hyphens(content)
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                changed.append(path)

# README too
for path in ['README.md']:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = fix_hyphens(content)
        if new_content != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            changed.append(path)

print(f'Fixed separator hyphens in {len(changed)} files:')
for p in sorted(changed):
    print(f'  {p}')

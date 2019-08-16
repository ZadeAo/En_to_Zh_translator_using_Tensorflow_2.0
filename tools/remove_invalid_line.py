import io
import re


def not_blank(s):
    if s.replace(' ', '') == '':
        return False
    else:
        return True


def has_chinese(s):
    chs = re.compile(u'[\u4e00-\u9fa5]')
    if chs.search(s):
        return True
    else:
        return False


def remove_invalid_line(old, new):
    lines = io.open(old, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('\t') for l in lines]
    new_pairs = [[s[0], s[1]] for s in pairs if len(s) == 2 and not_blank(
        s[0]) and not_blank(s[1]) and has_chinese(s[1])]
    new_lines = '\n'.join(['\t'.join(s) for s in new_pairs])

    with open(new, 'w') as f:
        f.write(new_lines)


if __name__ == "__main__":
    remove_invalid_line('2.tsv', '3.tsv')

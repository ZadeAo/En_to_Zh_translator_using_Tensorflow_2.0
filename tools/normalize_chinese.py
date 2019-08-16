import io
import re


def normalize(s):
    s = s.replace('(', '（')
    s = s.replace(')', '）')
    s = s.replace('!', '！')
    s = s.replace('?', '？')
    fil = re.compile(
        u'[^0-9a-zA-Z\u4e00-\u9fa5.,，。、：；【】 （）《》！%？—·~‘’“”]+', re.UNICODE)
    s = fil.sub('', s)
    s = s.strip()
    return s


def normalize_chinese(old, new):
    lines = io.open(old, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('\t') for l in lines]
    new_pairs = [[s[0], normalize(s[1])] for s in pairs]
    new_lines = '\n'.join(['\t'.join(s) for s in new_pairs])

    with open(new, 'w') as f:
        f.write(new_lines)


if __name__ == "__main__":
    normalize_chinese('1.tsv', '2.tsv')
    # print(normalize(' 这一代人的成就是  现代��史上最伟大的成就之一。f 发esjkl  ** 。，, d '))

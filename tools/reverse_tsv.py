import io


def reverse(old, new):
    lines = io.open(old, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('\t') for l in lines]
    new_pairs = [[s[1], s[0]] for s in pairs]
    new_lines = '\n'.join(['\t'.join(s) for s in new_pairs])

    with open(new, 'w') as f:
        f.write(new_lines)


if __name__ == "__main__":
    reverse('', '')

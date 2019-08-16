import io


def sort(old, new):
    lines = io.open(old, encoding='utf-8').read().strip().split('\n')
    lines.sort(key=lambda x: len(x))
    lines = '\n'.join(lines)

    with open(new, 'w') as f:
        f.write(lines)


if __name__ == "__main__":
    sort('3.tsv', '4.tsv')

import io
import re
import os


def normalize():
    config = re.findall(
        r'\w+ {0,1}=', io.open('config.py', 'r', encoding='utf-8').read())
    config = [re.findall(r'\w+', w)[0] for w in config]

    for file in os.listdir('./'):
        if os.path.splitext(file)[-1] == '.py' and os.path.splitext(file)[0] != 'config':
            f = io.open(file, 'r', encoding='utf-8')
            all_text = f.read()
            lines = all_text.split('\n')

            for line in lines:
                if line.startswith('from config import'):
                    f.close()
                    all_text = '\n'.join([l for l in lines if l != line])
                    os.remove(file)

                    used = []
                    for w in config:
                        if w in all_text:
                            used.append(w)
                    if used:
                        new = 'from config import ' + ', '.join(used)
                    else:
                        new = ''

                    new_text = '\n'.join(
                        [l if l != line else new for l in lines])

                    f = io.open(file, 'w', encoding='utf-8')
                    f.write(new_text)
                    f.close
                    break


if __name__ == "__main__":
    normalize()

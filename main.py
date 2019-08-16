from train import train
from apply import apply
from pathlib import Path
from config import MODEL_PATH
import os
import shutil
import pprint


def main(sentences, force_retrain=False):
    if force_retrain:
        if Path(MODEL_PATH).is_dir():
            shutil.rmtree(MODEL_PATH)
        os.mkdir(MODEL_PATH)

    if not Path(MODEL_PATH).is_dir():  # not exists
        os.mkdir(MODEL_PATH)

    if not os.listdir(MODEL_PATH):  # blank
        train()

    pprint.pprint(apply(sentences))


if __name__ == '__main__':
    s = [
        'I love you.',
        'I know you.',
        "It's me!",
        "That's my book.",
        'What is your name?',
        'Do you love me?',
        'How are you?',
        'I ate a hamburger.',
        'You are wonderful.',
        'What is the baby doing?',
        'I know what you mean.',
        'The elephant likes painting.',
        "I don't like you.",
        "What's your name?",
        'I like football.',
        'Experience is the best teacher.',
        'He is getting better day by day.',
        'I cannot agree with you on the matter.',
        'Please make three copies of each page.',
        'The wind blew too hard for them to play in the park.',
        'I am convinced that things will change for the better.',
        'At the time, There were no native English speakers teaching in any public school.',
        'Some people clung to tree branches for several hours to avoid being washed away by the floodwaters.'
    ]

    main(s)

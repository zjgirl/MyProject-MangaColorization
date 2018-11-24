
from cGanNetwork import Color

if __name__ == '__main__':

    print("please input train or sample?")

    cmd = input()

    if cmd == "train":

        c = Color()

        c.train()

    elif cmd == "sample":

        c = Color(512,1)

        c.sample()

    else:

        print ("Usage: python main.py [train, sample]")
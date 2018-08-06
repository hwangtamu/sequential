import requests, threading, os
from bs4 import BeautifulSoup
from fractions import Fraction
from itertools import permutations


class Data(object):
    def __init__(self):
        self.d = {
            'alt': '1',
            'fusing': '2',
            'alchemy': '3',
            'chaos': '4',
            'gcp': '5',
            'exalt': '6',
            'chrome': '7',
            'jeweler': '8',
            'Orb of Chance': '9',
            'chisel': '10',
            'Orb of Scouring':'11',
            'Blessed Orb': '12',
            'regret':'13',
            'regal': '14',
            'divine orb': '15',
            'vaal orb': '16',
            'Orb of Transmutation': '22',
            'Orb of Augmentation': '23',
            'Silver Coin': '35'
            # ,'Apprentice Cartographers Sextant': '45'
            # ,'Journeyman Cartographers Sextant': '46'
            # ,'Master Cartographers Sextant': '47'
        }

    def load(self):
        #TODO:
        # load database from file
        pass


class Monitor(object):
    def __init__(self):
        db = Data()
        self.data = db.d

    def search(self, a, b, league='Incursion'):
        attr = ['data-ign', 'data-sellvalue', 'data-buyvalue']
        if a not in self.data or b not in self.data:
            print(a, b)
            raise KeyError
        url = 'http://currency.poe.trade/search?league='+league+'&online=x&want='+self.data[a]+'&have='+self.data[b]
        r = requests.get(url)
        soup = BeautifulSoup(r.text, features="html5lib")
        results = [(eval(x[attr[1]]+'/'+x[attr[2]]),"@" + x[attr[0]] + " Hi, I'd like to buy your "
                    + str(int(eval(x[attr[1]]))) + " " + a + " for my "
                    + str(int(eval(x[attr[2]]))) + " " + b + " in Incursion.")
                   for x in soup.find_all("div", class_='displayoffer')]
        return results

    def find_profit_2(self, c='chaos', rate=1.04, num=5):
        threading.Timer(180.0, self.find_profit_2).start()
        os.system('cls' if os.name == 'nt' else 'clear')
        for i in self.data:
            if self.data[i]!=self.data[c]:
                buy = self.search(c, i)[:num]
                sell = self.search(i, c)[:num]
                if buy[0][0]*sell[0][0]>rate:
                    for j in range(num):
                        for k in range(num):
                            if buy[j][0]*sell[k][0]>rate:
                                print('\033[93m'+str(Fraction(buy[j][0]*sell[k][0]).limit_denominator())+','+str(buy[j][0]*sell[k][0])+'\033[0m')
                                print(buy[j][1])
                                print(sell[k][1])

    def find_profit_3(self, c='chaos', rate=1.0, num=3):
        other = {}
        for i in self.data:
            if i!=c:
                other[i] = self.data[i]

        curr = permutations(other.keys(), 2)
        for item in list(curr):
            act_1 = self.search(c, item[0])
            act_2 = self.search(item[0], item[1])
            act_3 = self.search(item[1], c)

            act_1 = act_1[:num] if len(act_1)>num else act_1
            act_2 = act_2[:num] if len(act_2)>num else act_2
            act_3 = act_3[:num] if len(act_3)>num else act_3
            if len(act_1)*len(act_2)*len(act_3)==0:
                break
            if act_1[0][0]*act_2[0][0]*act_3[0][0]>rate:
                for x in act_1:
                    for y in act_2:
                        for z in act_3:
                            p = x[0]*y[0]*z[0]
                            if p>rate:
                                print(Fraction(p).limit_denominator(), p)
                                print(x[1])
                                print(y[1])
                                print(z[1])


if __name__=="__main__":
    m = Monitor()
    m.find_profit_2(rate=1.04)

totaldata = 570
ponerotounis = 60
bistotirs = 15
ektirisup = 25

ponerotounis = totaldata * (ponerotounis/100)
bistotirs = totaldata * (bistotirs/100)
ektirisup = totaldata * (ektirisup/100)
print(ponerotounis, bistotirs, ektirisup)

percent = int(input('Enter disease percent : '))


posponerotounis = ponerotounis * (percent/100)
posbistotirs = bistotirs * (percent/100)
posektirisup = ektirisup * (percent/100)
print(int(posponerotounis), int(posbistotirs), int(posektirisup))




# coding: utf-8

from conceptnet import ConceptNet

cn = ConceptNet(filepath="data/russian-conceptnet.pickle.bz2")

print(cn.check_pair("собака", "пёс"))
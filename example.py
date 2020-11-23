# -*- coding: utf-8 -*-

from ruconceptnet import ConceptNet

cn = ConceptNet(filepath="ruconceptnet/data/russian-conceptnet.pickle.bz2")

print(cn.check_pair("собака", "пёс"))
import pickle
import urllib

f = open('../../data/urls/ILSVRC2010_urls.txt', 'r')
data, class_list, class_meta = pickle.load(f)
f.close()

url=[d[0] for d in data]

for i in xrange(len(url)):
    try:
        f=urllib.urlopen(url[i])
        print((i,f.info().type))
    except:
        pass
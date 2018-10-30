import sys
import os

def wget( address, target ):
    print('wget ' + address + ' -O ' + target)
    version = sys.version_info
    if not os.path.exists("./data"):
        os.mkdir("./data")
    if version[0] == 2:
        import urllib
        urllib.urlretrieve(address, target)
    elif version[0] == 3:
        import urllib.request
        urllib.request.urlretrieve(address,  target)

wget('https://preferredjp.box.com/shared/static/o2xip23e3f0knwc5ve78oderuglkf2wt.onnx', './data/vgg16.onnx')
wget('https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt', './data/synset_words.txt')
wget('https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg', './data/Light_sussex_hen.jpg')


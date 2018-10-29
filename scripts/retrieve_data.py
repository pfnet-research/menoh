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

wget('https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1', './data/VGG16.onnx')
wget('https://preferredjp.box.com/shared/static/x4k1pya1w1vuvpfk77pp6ma8oaye8syb.onnx', './data/resnet50.onnx')
wget('https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt', './data/synset_words.txt')
wget('https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg', './data/Light_sussex_hen.jpg')


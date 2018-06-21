import sys

def wget( address, target ):
    print('wget ' + address + ' -O ' + target)
    version = sys.version_info
    if version[0] == 2:
        import urllib
        urllib.urlretrieve(address, target)
    elif version[0] == 3:
        import urllib.request
        urllib.request.urlretrieve(address,  target)

wget('https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1', './data/VGG16.onnx')
wget('https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt', './data/synset_words.txt')
wget('https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg', './data/Light_sussex_hen.jpg')


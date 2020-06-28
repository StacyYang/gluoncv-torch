"""Model store which provides pretrained models."""
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('7591854d34e97010f019e9f98f9aed9c4a463d58', 'resnet18'),
    ('64557eb2096a56ff2db4fdd637e00ede804c1fec', 'resnet34'),
    ('0ef8ed2db4162747fecb34ee542b944d141b3ef1', 'resnet50'),
    ('1834038c51dd60b2819b4391acfec0bb4525f986', 'resnet101'),
    ('990926f3c93c67aea2342d1a5b88ba63dfee32f4', 'resnet152'),
    ('357fb3777da3ebdf13ab06bee51fa6f83837967c', 'fcn_resnet101_voc'),
    ('8bb3bccd02da0e5431a616d3abe7e8c383e8f587', 'fcn_resnet101_ade'),
    ('6d90aaae73a3adcb20f186895b27bf45368601ab', 'psp_resnet101_voc'),
    ('fe990f00dda51d58718c43cf4705e0a61ca15ef0', 'psp_resnet101_ade'),
    ('5c25b7db003fb805df6574139bf04e1b85f0f37d', 'deeplab_resnet101_voc'),
    ('c0d88de54f3abbc358038c248f0863bef96fb0d4', 'deeplab_resnet101_ade'),
    ]}

gluoncvth_repo_url = 'https://s3.us-west-1.wasabisys.com/resnest'
_url_format = '{repo_url}gluoncvth/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.gluoncvth', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.gluoncvth/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.pth')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file {} detected.' +
                  ' Downloading again.'.format(file_path))
    else:
        print('Model file {} is not found. Downloading.'.format(file_path))

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = os.environ.get('ENCODING_REPO', gluoncvth_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join('~', '.gluoncvth', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.gluoncvth/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())

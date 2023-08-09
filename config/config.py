from yacs.config import CfgNode as CN


_C = CN()

_C.DEVICE_INDEX = None
_C.SPLITS = None
_C.ADD_NEW_TOKENS = None
_C.LR = None

_C.DATA = CN()
_C.DATA.ADD_DENSE = None
_C.DATA.BATCH_SIZE = None
_C.DATA.N_EPOCHS = None
_C.DATA.TEST_EVERY = None
_C.DATA.SAVE_EVERY = None

_C.DATA.args = CN()
_C.DATA.args.root = None
_C.DATA.args.train_pct = None
_C.DATA.args.valid_pct = None

_C.MODEL = CN()
_C.MODEL.HF_PATH = None
_C.MODEL.MAX_LENGTH = None
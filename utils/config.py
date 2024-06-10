import mmengine
from mmengine.utils import import_modules_from_strings
class Config(mmengine.Config):

    def __init__(self, 
        filename=None, cfg_dict = None, 
        cfg_text=None, text_format=None,
        use_predefined_variables=True, 
        import_custom_modules=True
    ):
        
        if cfg_dict is None: cfg_dict = dict()
        if filename is not None:
            cfg_dict = super().fromfile(filename,
                use_predefined_variables, import_custom_modules)._cfg_dict
        if cfg_text is not None:
            assert text_format is not None
            # print("cfg_text needs text_format.")
            cfg_dict = super().fromstring(cfg_text, text_format)._cfg_dict

        super(Config, self).__init__(
            cfg_dict=cfg_dict, cfg_text=cfg_text, filename=filename)

        # init to import modules
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            try:
                import_modules_from_strings(**cfg_dict['custom_imports'])
            except ImportError as e:
                raise ImportError('Failed to custom import!') from e


    def hasKey(self, name: str) -> bool:
        names = name.split('.') if name.split('.') is not None else [name]
        _temp_dict = self._cfg_dict.copy()
        check_keys = _temp_dict.keys()
        for n in names:
            if n in check_keys:
                _temp_dict = _temp_dict[n]
                if isinstance(_temp_dict, dict):
                    check_keys = _temp_dict.keys()
                else:
                    if names[-1] != n: return False
                continue
            return False
        return True


    # @staticmethod
    # def fromfile(filename, use_predefined_variables=True):
    #     if isinstance(filename, Path):
    #         filename = str(filename)
    #     cfg_dict, cfg_text = Config._file2dict(filename,
    #                                            use_predefined_variables)
    #     return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    # @staticmethod
    # def fromstring(cfg_str, file_format):
    #     from pathlib import Path
    #     import warnings
    #     import os
    #     import tempfile
    #     if file_format not in ['.py', '.json', '.yaml', '.yml']:
    #         raise OSError('Only py/yml/yaml/json type are supported now!')
    #     if file_format != '.py' and 'dict(' in cfg_str:
    #         # check if users specify a wrong suffix for python
    #         warnings.warn(
    #             'Please check "file_format", the file format may be .py')
    #     with tempfile.NamedTemporaryFile(
    #             'w', encoding='utf-8', suffix=file_format,
    #             delete=False) as temp_file:
    #         temp_file.write(cfg_str)
    #         # on windows, previous implementation cause error
    #         # see PR 1077 for details
    #     cfg = Config.fromfile(temp_file.name)
    #     os.remove(temp_file.name)
    #     return cfg
        
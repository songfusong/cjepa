from src.third_party.videosaur.videosaur.data.utils import get_data_root_dir


def build(*args, **kwargs):
    from src.third_party.videosaur.videosaur.data.datamodules import build as build_datamodule

    return build_datamodule(*args, **kwargs)


__all__ = ["build", "get_data_root_dir"]

import argparse

from .translate_tr import translate


if __name__ == "__main__":
    argparse.ArgumentParser(description="Türkçe veri kümesini İngilizceye çevir.").parse_args()
    translate()

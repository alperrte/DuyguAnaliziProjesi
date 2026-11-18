import argparse

from .vader_analysis import vader


if __name__ == "__main__":
    argparse.ArgumentParser(description="VADER ile duygu analizi yap.").parse_args()
    vader()

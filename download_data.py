"""Download the IT'IS Tissue Properties Database V5.0 and extract the table used here.

The IT'IS database is distributed by the IT'IS Foundation under its own terms. This
script downloads it directly from the official source. It is not redistributed in
this repository. See https://itis.swiss/database for license and citation details.
"""
import io
import os
import sys
import urllib.request
import zipfile

URL = "https://itis.swiss/assets/Downloads/TissueDb/Database-V5-0.zip"
TARGET = "Thermal_dielectric_acoustic_MR properties_database_V5.0(Excel).xls"


def main():
    if os.path.exists(TARGET):
        print(f"{TARGET} already present, nothing to do.")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        data = resp.read()
    print(f"Downloaded {len(data)} bytes. Extracting {TARGET} ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        match = [n for n in names if n.endswith(TARGET)]
        if not match:
            print("Could not find the expected table in the archive. Files present:")
            for n in names:
                print("  ", n)
            sys.exit(1)
        with zf.open(match[0]) as src, open(TARGET, "wb") as dst:
            dst.write(src.read())
    print(f"Wrote {TARGET}.")


if __name__ == "__main__":
    main()

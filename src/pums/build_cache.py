"""Convert the raw PUMS CSVs into a columnar parquet cache.

Every column is stored as a *string*, exactly as it appears in the CSV. PUMS
uses blanks for "outside the universe", and letting a CSV reader guess types
silently turns those blanks into 0.0 or coerces keys like SERIALNO into
floats. Reading as text and converting explicitly, per question, keeps the
universe rules visible in the code that depends on them.

Conversion streams in record batches so peak memory stays small.
"""
from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .config import CACHE, RAW, cache_path


def convert(kind: str, year: int) -> None:
    src = RAW[(kind, year)]
    dst = cache_path(kind, year)
    if not src.exists():
        raise FileNotFoundError(f"missing raw file: {src}")

    read_opts = pacsv.ReadOptions(block_size=1 << 24)  # 16 MiB blocks
    # strings_can_be_null=False keeps "" as an empty string rather than null,
    # so "blank" stays distinguishable from a genuinely absent field.
    convert_opts = pacsv.ConvertOptions(strings_can_be_null=False)

    reader = pacsv.open_csv(src, read_options=read_opts, convert_options=convert_opts)
    # Force every column to string up front so the schema never shifts between
    # batches (a column that looks numeric early can hit a blank later).
    schema = pa.schema([pa.field(n, pa.string()) for n in reader.schema.names])

    writer = pq.ParquetWriter(dst, schema, compression="zstd")
    rows = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch]).cast(schema)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        writer.close()
        reader.close()

    print(f"  {kind} {year}: {rows:,} rows, {len(schema.names)} cols -> {dst.name}")


def main() -> int:
    CACHE.mkdir(exist_ok=True)
    print("Building parquet cache...")
    for (kind, year) in RAW:
        convert(kind, year)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

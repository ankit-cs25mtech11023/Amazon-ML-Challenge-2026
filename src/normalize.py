"""Stage 0: normalize names and addresses for all sources.

Reads raw TSVs, writes work/norm/{split}_{source}.parquet with cleaned fields.
Uses only the provided data plus hand-written abbreviation tables.
"""
import os
import re
import sys
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

DATA = os.environ.get("DATA_DIR", "student_resource/dataset")
OUT = os.environ.get("NORM_DIR", "work/norm")

# Legal / entity-form words (removed from the "core" name, kept as a separate form field)
LEGAL = {
    "the", "inc", "incorporated", "llc", "l", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "plc", "llp", "lp", "pllc", "pc", "pa", "sarl", "sas", "sasu", "eurl", "sa", "sci",
    "snc", "cie", "et", "and", "of", "pte", "opc", "dba", "ei", "scop", "sca", "selarl", "sel", "ets", "etablissements",
}
LEGAL_FORM = {
    "pvt": "private", "private": "private", "ltd": "limited", "limited": "limited", "llc": "llc", "inc": "inc",
    "incorporated": "inc", "corp": "corp", "corporation": "corp", "llp": "llp", "co": "co", "company": "co",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "eurl": "eurl", "sa": "sa", "sci": "sci", "plc": "plc",
}
# Address token canonicalization (US / India / France)
ADDR_MAP = {
    # US
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "crt": "court", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "cir": "circle",
    "pl": "place", "sq": "square", "ter": "terrace", "trl": "trail", "tpke": "turnpike", "ste": "suite",
    "apt": "unit", "apartment": "unit", "suite": "unit", "fl": "floor", "flr": "floor", "mt": "mount", "ft": "fort", "pt": "point", "hts": "heights", "jct": "junction",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    # India
    "no": "no", "h": "house", "hno": "house", "ngr": "nagar", "cly": "colony", "col": "colony", "nr": "near",
    "opp": "opposite", "bldg": "building", "mkt": "market", "stn": "station", "dist": "district",
    "distt": "district", "tq": "taluk", "tal": "taluk", "vill": "village", "vil": "village", "po": "post",
    "gf": "ground", "ff": "1st", "sf": "2nd", "mg": "mg", "sec": "sector", "sect": "sector",
    "extn": "extension", "ext": "extension", "indl": "industrial", "ind": "industrial", "estt": "estate",
    "chowk": "chowk", "marg": "marg", "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "gurgaon": "gurugram", "allahabad": "prayagraj",
    # France
    "r": "rue", "imp": "impasse", "all": "allee", "chem": "chemin", "rte": "route", "qu": "quai",
    "crs": "cours", "sq": "square", "bis": "bis", "ter": "ter", "lot": "lotissement", "res": "residence",
    "zi": "zone", "za": "zone", "zac": "zone", "ch": "chemin", "fbg": "faubourg", "prom": "promenade",
}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "jk": "jammu kashmir", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "od": "odisha", "or": "odisha", "orissa": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
    "tn": "tamil nadu", "ts": "telangana", "tg": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal", "dl": "delhi", "ch": "chandigarh",
    "py": "puducherry", "la": "ladakh",
}
STATE_NAMES = set(US_STATES.values()) | set(IN_STATES.values())
FR_REGIONS = {
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire", "corse",
    "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine", "occitanie",
    "pays de la loire", "provence alpes cote d azur",
}

# France: region / department names are implied by the city and vary between sources -> drop them
FR_DROP = r" (hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord|france)(?: (hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord|france))*"
FR_DROP = FR_DROP + " "
_rep = re.compile(r"(.)\1+")
_nonalnum = re.compile(r"[^a-z0-9]+")


def to_latin(s: str) -> str:
    """Transliterate any script to ASCII, lowercase, collapse repeated letters from Indic romanization."""
    if s is None:
        return ""
    if s.isascii():
        return s.lower()
    t = unidecode(s).lower()
    return t


def fold_repeats(s: str) -> str:
    return _rep.sub(r"\1", s)


def _latin_batch(vals):
    out = []
    for v in vals:
        if v is None:
            out.append("")
        elif v.isascii():
            out.append(v.lower())
        else:
            out.append(unidecode(v).lower())
    return out


def latinize(series: pl.Series, procs: int) -> pl.Series:
    vals = series.to_list()
    n = len(vals)
    chunk = max(1, n // (procs * 4))
    parts = [vals[i:i + chunk] for i in range(0, n, chunk)]
    with Pool(procs) as p:
        res = p.map(_latin_batch, parts)
    return pl.Series(series.name, [x for r in res for x in r], dtype=pl.Utf8)


def soundex_tok(t: str) -> str:
    """Cheap consonant-class skeleton (robust to vowel/schwa noise from transliteration)."""
    if not t:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
             "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    out = [t[0]]
    last = codes.get(t[0], "")
    for ch in t[1:]:
        c = codes.get(ch, "")
        if c and c != last:
            out.append(c)
        last = c
    return "".join(out)[:5]


def build(df: pl.DataFrame, procs: int) -> pl.DataFrame:
    src = df["entity_id"].str.slice(0, 2)
    name_raw = latinize(df["business_name"], procs)
    addr_raw = latinize(df["business_address"], procs)
    df = df.with_columns(src.alias("src"), name_raw.alias("name_l"), addr_raw.alias("addr_l"))

    legal_list = list(LEGAL)
    df = df.with_columns(
        # garbled name kinds
        pl.col("name_l").str.contains(r"\.(com|in|net|org|co|fr|io|biz)\b|www\.").alias("name_is_url"),
        pl.col("name_l").str.contains(r"^\s*[@#]").alias("name_is_handle"),
        (~pl.col("business_name").str.contains(r"^[\x00-\x7F\p{Latin}\p{M}\p{P}\p{S}\p{N}\s]*$")).alias("name_nonlatin"),
        # strip urls / handles / phone numbers / bracket content
        pl.col("name_l")
        .str.replace_all(r"www\.", " ")
        .str.replace_all(r"\.(com|in|net|org|co|fr|io|biz)\b", " ")
        .str.replace_all(r"\+?\d[\d\- ]{7,}\d", " ")
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
        .alias("name_n"),
        pl.col("addr_l")
        .str.replace_all(r"\bn\s*deg\s*", " ")
        .str.replace_all(r"[^a-z0-9/]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"/", " / ")
        .str.strip_chars()
        .alias("addr_n0"),
    )
    # full state names -> abbreviation (single canonical token), per country
    a = pl.lit(" ") + pl.col("addr_n0") + pl.lit(" ")
    us, ind = a, a
    for ab, full in US_STATES.items():
        us = us.str.replace_all(" " + full + " ", " " + ab + " ", literal=True)
    for ab, full in IN_STATES.items():
        if ab in ("ct", "or", "ut", "tg"):
            continue
        ind = ind.str.replace_all(" " + full + " ", " " + ab + " ", literal=True)
    df = df.with_columns(
        pl.when(pl.col("country") == "US").then(us)
        .when(pl.col("country") == "India").then(ind.str.replace_all(" (orissa|odisha) ", " od ").str.replace_all(" (chhattisgarh) ", " cg ").str.replace_all(" (uttarakhand|uttaranchal) ", " uk ").str.replace_all(" (telangana) ", " ts "))
        .when(pl.col("country") == "France").then(
            a.str.replace_all(r" st e ", " sainte ").str.replace_all(r" ste ", " sainte ").str.replace_all(r" st ", " saint ")
            .str.replace_all(FR_DROP, " "))
        .otherwise(a).str.replace_all(r"\s+", " ").str.strip_chars().alias("addr_n0")
    )
    df = df.with_columns(
        pl.col("name_n").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("name_tok_all"),
        pl.col("addr_n0").str.split(" ").list.eval(
            pl.element().filter((pl.element() != "") & (pl.element() != "/"))
            .str.replace(r"^0+(\d)", "${1}")
            .replace(ADDR_MAP)
        ).alias("addr_tok"),
    )
    df = df.with_columns(
        pl.col("name_tok_all").list.eval(pl.element().filter(~pl.element().is_in(legal_list))).alias("name_tok"),
        pl.col("name_tok_all").list.eval(pl.element().filter(pl.element().is_in(list(LEGAL_FORM))).replace(LEGAL_FORM)).list.unique().list.sort().list.join(" ").alias("legal_form"),
        pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+[a-z]?$"))).list.unique().alias("addr_num"),
        pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d{6}$"))).list.unique().alias("postal"),
        pl.col("addr_tok").list.join(" ").alias("addr_n"),
    )
    df = df.with_columns(
        pl.col("name_tok").list.join(" ").alias("name_core"),
        pl.col("name_tok").list.join("").alias("name_ns"),
        pl.col("name_tok").list.sort().list.join(" ").alias("name_key"),
        pl.col("name_tok").list.eval(pl.element().map_elements(lambda t: soundex_tok(fold_repeats(t)), return_dtype=pl.Utf8)).alias("name_sdx"),
        (pl.col("business_address").str.strip_chars() == "").alias("addr_empty"),
        pl.col("addr_tok").list.len().alias("addr_ntok"),
        pl.col("name_tok").list.len().alias("name_ntok"),
    )
    return df.drop("name_l", "addr_l", "addr_n0", "name_tok_all")


def main():
    procs = int(os.environ.get("PROCS", "48"))
    os.makedirs(OUT, exist_ok=True)
    jobs = [("train", s) for s in ("source1", "source2", "source3")] + [("test", s) for s in ("source1", "source2", "source3")]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if f"{j[0]}_{j[1]}" in sys.argv[1:]]
    for split, s in jobs:
        df = pl.read_csv(f"{DATA}/{split}/{split}_{s}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
        df = df.with_columns(pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""))
        out = build(df, procs)
        out.write_parquet(f"{OUT}/{split}_{s}.parquet")
        print(split, s, out.shape, flush=True)


if __name__ == "__main__":
    main()
